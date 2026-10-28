//! Reconstruction of milled mesh parts: every face is a wall parallel to one of a few
//! machining directions or a floor across one (pockets, steps, bosses, through and blind
//! holes from several sides, rounded corners, chamfers along an edge direction).
//!
//! Each direction contributes a stack of extruded *shadows*: between two floor levels the
//! shadow is the union of the part's sections, so the stack contains the part and is exact
//! for every feature cut along that direction. The part is the intersection of the stacks.
//! The result is certified against the mesh (volume and inside/outside probes on both sides
//! of every triangle) and refused when it deviates, never returned approximately.

use std::collections::BTreeMap;

use solidcraft_geom::{Loop2, Plane, Region2, Seg2, Vec2, Vec3};

use crate::meshbody::TriMesh;
use crate::{Body, BoolOp, KernelError, Result};

/// Machining directions tried at most (each is one stack of extrusions and one boolean).
const MAX_DIRECTIONS: usize = 6;
/// Candidate directions scored when choosing them.
const MAX_CANDIDATES: usize = 48;
/// Floor levels along one direction.
const MAX_LEVELS: usize = 64;
/// Sections whose union makes one slab's shadow.
const MAX_SAMPLES: usize = 12;
/// Points of one section loop that are fitted with lines and arcs (longer loops stay polygons).
const MAX_FIT_POINTS: usize = 4096;
/// Section segments per slab sample (bounds the planar booleans).
const MAX_SECTION_POINTS: usize = 60_000;

fn invalid(message: &str) -> KernelError {
    KernelError::Invalid(format!("mesh conversion: {message}"))
}

struct Tri {
    v: [u32; 3],
    n: Vec3,
    area: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    /// Parallel to the direction: part of a constant section.
    Wall,
    /// Across the direction at one height: a floor or ceiling level.
    Floor,
    Other,
}

/// A milled part as analytic planes and cylinders, or an error saying why it is not one.
/// `mesh` is welded, closed and consistently oriented.
/// Surfaces recognised beforehand that no machining direction makes: drill points and
/// spherical pockets, already filled in the mesh to be milled. Each is cut from the milled
/// solid as a solid of its own.
#[derive(Default)]
pub(crate) struct Features {
    /// The surface, the vertices of its region, and whether the region is concave.
    pub(crate) solids: Vec<(crate::Primitive, Vec<Vec3>, bool)>,
}

/// A milled part from `mesh` (possibly with its vertices put onto recognised surfaces),
/// certified against the `original` mesh.
pub(crate) fn convert_milled_with(mesh: &TriMesh, original: &TriMesh, tolerance: f64, features: &Features) -> Result<Body> {
    let bounds = mesh.positions.iter().fold(solidcraft_geom::Aabb3::EMPTY, |mut b, p| {
        b.add(*p);
        b
    });
    let size = bounds.size().len().max(1.);
    let epsilon = (size * 1e-7).min(tolerance * 0.01).max(1e-9);
    // Deviation within which a triangle counts as a wall or a floor: a fraction of the
    // conversion tolerance, never below the coordinate noise of an STL (f32) file.
    let snap = (tolerance * 0.1).max(epsilon * 3.);
    let tris = triangles(mesh)?;
    let total: f64 = tris.iter().map(|t| t.area).sum();
    if !(total > 0.) {
        return Err(invalid("empty mesh"));
    }
    let candidates = candidate_directions(&tris);
    let directions = choose_directions(mesh, &tris, &candidates, snap, total)?;
    // The first direction's stack is the stock; every other direction cuts it like a setup
    // on a machine: away whatever lies outside its outline, then its pockets. Cuts rather than
    // intersections keep the outer faces the directions share from meeting face to face.
    let mut result: Option<Body> = None;
    for d in &directions {
        // Sketch axes along another machining direction square to this one, so the outline's
        // walls run along the plane's axes.
        let across = directions.iter().copied().find(|e| e.dot(*d).abs() < 1e-6);
        let plane = across.and_then(|e| Plane::new(Vec3::ZERO, e, d.cross(e))).or_else(|| Plane::from_normal(Vec3::ZERO, *d));
        let plane = plane.ok_or_else(|| invalid("degenerate machining direction"))?;
        let stack = direction_stack(mesh, &tris, plane, tolerance, epsilon, snap, size)?;
        let btol = (snap * 0.5).clamp(1e-9, 1.);
        let mut body = match result {
            None => crate::extrude_planar_regions(&stack.plane, &stack.outline.iter().map(sealed).collect::<Vec<_>>(), stack.lo, stack.hi)?,
            Some(r) => {
                let mut r = r;
                let outside: Vec<Region2> = planar(&[bounding_rectangle(&stack.outline, snap)], &stack.outline, BoolOp::Cut, btol)?
                    .into_iter()
                    .filter(|c| !is_sliver(c, tolerance))
                    .collect();
                // One cut per piece: pieces may touch at a corner, which one extrusion refuses.
                for piece in &outside {
                    let tool = crate::extrude_planar_regions(&stack.plane, &[sealed(piece)], stack.lo - stack.margin, stack.hi + stack.margin)?;
                    r = crate::boolean(&r, &tool, BoolOp::Cut)?.ok_or_else(|| invalid("the machining directions leave no material"))?;
                }
                r
            }
        };
        for (a, b, regions) in &stack.tools {
            for piece in regions {
                let tool = crate::extrude_planar_regions(&stack.plane, &[sealed(piece)], *a, *b)?;
                body = crate::boolean(&body, &tool, BoolOp::Cut)?.ok_or_else(|| invalid("a pocket removes the whole part"))?;
            }
        }
        result = Some(body);
    }
    let mut result = result.ok_or_else(|| invalid("no machining direction found"))?;
    for (surface, vertices, concave) in &features.solids {
        if !concave {
            return Err(invalid(
                "raised cones and spheres (a domed boss, a pointed pin) are not supported yet; only drill points and spherical pockets",
            ));
        }
        let tool = feature_solid(surface, vertices)?;
        result = crate::boolean(&result, &tool, BoolOp::Cut)?.ok_or_else(|| invalid("a recognised pocket removes the whole part"))?;
    }
    let original_tris = triangles(original)?;
    let original_total: f64 = original_tris.iter().map(|t| t.area).sum();
    certify(&result, original, &original_tris, tolerance, original_total, epsilon, size)?;
    Ok(result)
}

/// The solid a recognised concave cone or sphere cuts: a cone from its apex to the farthest of
/// its region's vertices, or the whole sphere.
pub(crate) fn feature_solid(surface: &crate::Primitive, vertices: &[Vec3]) -> Result<Body> {
    match *surface {
        crate::Primitive::Sphere { center, radius } => crate::sphere(center, radius),
        crate::Primitive::Cone { apex, axis, half_angle } => {
            let depth = vertices.iter().map(|p| (*p - apex).dot(axis)).fold(0f64, f64::max);
            if !(depth > 0.) {
                return Err(invalid("a recognised cone has no depth"));
            }
            let across = axis.any_perp().normalized().ok_or_else(|| invalid("degenerate cone axis"))?;
            let plane = Plane::new(apex, across, axis).ok_or_else(|| invalid("degenerate cone axis"))?;
            let profile = Region2 {
                outer: Loop2::polygon(&[Vec2::new(0., 0.), Vec2::new(depth * half_angle.tan(), depth), Vec2::new(0., depth)]),
                holes: Vec::new(),
            };
            crate::revolve(&plane, &[profile], Vec2::new(0., 0.), Vec2::new(0., 1.), std::f64::consts::TAU)?
                .into_iter()
                .next()
                .ok_or_else(|| invalid("could not build the recognised cone"))
        }
        _ => Err(invalid("only cones and spheres are cut as recognised features")),
    }
}

fn triangles(mesh: &TriMesh) -> Result<Vec<Tri>> {
    mesh.triangles
        .iter()
        .map(|t| {
            let [a, b, c] = *t;
            let (Some(pa), Some(pb), Some(pc)) = (mesh.positions.get(a as usize), mesh.positions.get(b as usize), mesh.positions.get(c as usize))
            else {
                return Err(invalid("triangle refers to a missing vertex"));
            };
            let cross = (*pb - *pa).cross(*pc - *pa);
            // A zero-area sliver keeps the mesh closed but faces nowhere.
            let n = cross.normalized().unwrap_or(Vec3::ZERO);
            Ok(Tri { v: *t, n, area: cross.len() * 0.5 })
        })
        .collect()
}

fn point(mesh: &TriMesh, i: u32) -> Vec3 {
    mesh.positions.get(i as usize).copied().unwrap_or(Vec3::ZERO)
}

/// One sign per line: the direction with a positive leading coordinate.
fn canonical(d: Vec3) -> Vec3 {
    if d.x < -1e-9 || (d.x.abs() <= 1e-9 && (d.y < -1e-9 || (d.y.abs() <= 1e-9 && d.z < 0.))) { -d } else { d }
}

/// Directions worth trying, best first: face normals (floors) and the lines where
/// neighbouring faces meet (walls; the facets of a cylinder meet along its axis).
fn candidate_directions(tris: &[Tri]) -> Vec<Vec3> {
    let mut bins = BTreeMap::<(i64, i64, i64), (Vec3, f64)>::new();
    let mut add = |d: Vec3, w: f64| {
        let d = canonical(d);
        let key = ((d.x * 400.).round() as i64, (d.y * 400.).round() as i64, (d.z * 400.).round() as i64);
        let entry = bins.entry(key).or_insert((Vec3::ZERO, 0.));
        entry.0 += d * w;
        entry.1 += w;
    };
    let mut sides = BTreeMap::<(u32, u32), Vec<usize>>::new();
    for (i, t) in tris.iter().enumerate() {
        add(t.n, t.area);
        let [a, b, c] = t.v;
        for (a, b) in [(a, b), (b, c), (c, a)] {
            sides.entry((a.min(b), a.max(b))).or_default().push(i);
        }
    }
    for pair in sides.values() {
        let [i, j] = pair.as_slice() else { continue };
        let (Some(a), Some(b)) = (tris.get(*i), tris.get(*j)) else { continue };
        let c = a.n.cross(b.n);
        if c.len() > 0.004
            && let Some(c) = c.normalized()
        {
            add(c, a.area.min(b.area));
        }
    }
    let mut bins: Vec<(Vec3, f64)> = bins.into_values().filter_map(|(sum, w)| Some((sum.normalized()?, w))).collect();
    bins.sort_by(|a, b| b.1.total_cmp(&a.1));
    let close = 1f64.to_radians().cos();
    let mut clusters: Vec<(Vec3, f64)> = Vec::new();
    for (d, w) in bins.into_iter().take(20_000) {
        if let Some(c) = clusters.iter_mut().find(|c| c.0.normalized().is_some_and(|cd| cd.dot(d).abs() > close)) {
            let sign = if c.0.dot(d) < 0. { -1. } else { 1. };
            c.0 += d * (w * sign);
            c.1 += w;
        } else if clusters.len() < 256 {
            clusters.push((d * w, w));
        }
    }
    clusters.sort_by(|a, b| b.1.total_cmp(&a.1));
    // Model axes stay exact: an STL of an axis-aligned part keeps its planes axis-aligned.
    let snap_axis = |d: Vec3| {
        let near = 0.02f64.to_radians().cos();
        [Vec3::X, Vec3::Y, Vec3::Z].into_iter().find(|a| d.dot(*a).abs() > near).unwrap_or(d)
    };
    let mut out: Vec<Vec3> = Vec::new();
    for (sum, _) in clusters {
        let Some(d) = sum.normalized() else { continue };
        let d = canonical(snap_axis(d));
        if !out.iter().any(|o| o.dot(d).abs() > close) {
            out.push(d);
        }
        if out.len() >= MAX_CANDIDATES {
            break;
        }
    }
    out
}

fn heights(mesh: &TriMesh, t: &Tri, d: Vec3) -> [f64; 3] {
    let [a, b, c] = t.v;
    [point(mesh, a).dot(d), point(mesh, b).dot(d), point(mesh, c).dot(d)]
}

/// How a triangle lies relative to direction `d`, within `snap`.
fn role(mesh: &TriMesh, t: &Tri, d: Vec3, snap: f64) -> Role {
    let h = heights(mesh, t, d);
    let (lo, hi) = (h[0].min(h[1]).min(h[2]), h[0].max(h[1]).max(h[2]));
    if hi - lo <= snap {
        return Role::Floor;
    }
    let [a, b, c] = t.v;
    let (pa, pb, pc) = (point(mesh, a), point(mesh, b), point(mesh, c));
    let longest = pa.dist(pb).max(pb.dist(pc)).max(pc.dist(pa));
    // Twice the projected area over the longest side: the projected triangle's height.
    let projected = (pb - pa).cross(pc - pa).dot(d).abs();
    if projected <= snap * longest { Role::Wall } else { Role::Other }
}

/// The fewest directions (greedily) that explain every triangle as a wall or a floor.
fn choose_directions(mesh: &TriMesh, tris: &[Tri], candidates: &[Vec3], snap: f64, total: f64) -> Result<Vec<Vec3>> {
    let roles: Vec<Vec<bool>> = candidates.iter().map(|d| tris.iter().map(|t| role(mesh, t, *d, snap) != Role::Other).collect()).collect();
    let mut explained = vec![false; tris.len()];
    let mut chosen: Vec<Vec3> = Vec::new();
    let mut used = vec![false; candidates.len()];
    while chosen.len() < MAX_DIRECTIONS {
        let mut best: Option<(usize, f64)> = None;
        for (c, r) in roles.iter().enumerate() {
            if used.get(c).copied().unwrap_or(true) {
                continue;
            }
            let gain: f64 = tris.iter().zip(r).zip(&explained).filter(|((_, ok), done)| **ok && !**done).map(|((t, _), _)| t.area).sum();
            if best.is_none_or(|(_, g)| gain > g) {
                best = Some((c, gain));
            }
        }
        let Some((c, gain)) = best else { break };
        if gain <= total * 1e-9 {
            break;
        }
        if let Some(u) = used.get_mut(c) {
            *u = true;
        }
        if let (Some(d), Some(r)) = (candidates.get(c), roles.get(c)) {
            chosen.push(*d);
            for (e, ok) in explained.iter_mut().zip(r) {
                *e |= *ok;
            }
        }
    }
    let unexplained: f64 = tris.iter().zip(&explained).filter(|(_, e)| !**e).map(|(t, _)| t.area).sum();
    if unexplained > total * 1e-4 {
        return Err(invalid(&format!(
            "{:.1}% of the surface is not parallel or square to {} machining directions (drill points, spheres, sloped or freeform faces); only milled and constant-section parts convert",
            unexplained / total * 100.,
            chosen.len().max(1)
        )));
    }
    if chosen.is_empty() {
        return Err(invalid("no machining direction found"));
    }
    Ok(chosen)
}

/// What one machining direction contributes: the outline of the part seen along it (the union
/// of its sections) and the pockets open to either end, as extrusions between heights along it.
/// Extruding the outline and cutting the pockets gives a solid containing the part, exact for
/// the features cut along the direction.
struct Stack {
    plane: Plane,
    lo: f64,
    hi: f64,
    margin: f64,
    outline: Vec<Region2>,
    /// (from, to, regions), deepest first.
    tools: Vec<(f64, f64, Vec<Region2>)>,
}

/// The axis-aligned rectangle (in the plane) around `regions` (arcs sampled within `tol`).
fn bounding_rectangle(regions: &[Region2], tol: f64) -> Region2 {
    let mut b = solidcraft_geom::Aabb2::EMPTY;
    for r in regions {
        for p in loop_points(&r.outer, tol) {
            b.add(p);
        }
    }
    let (lo, hi) = (b.min, b.max);
    Region2 { outer: Loop2::polygon(&[lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)]), holes: Vec::new() }
}

fn direction_stack(mesh: &TriMesh, tris: &[Tri], plane: Plane, tolerance: f64, epsilon: f64, snap: f64, size: f64) -> Result<Stack> {
    let d = plane.normal();
    let (lo, hi) = mesh.positions.iter().map(|p| p.dot(d)).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), h| (lo.min(h), hi.max(h)));
    if !(hi - lo > tolerance) {
        return Err(invalid("the part is thinner than the tolerance"));
    }
    let roles: Vec<Role> = tris.iter().map(|t| role(mesh, t, d, snap)).collect();
    let ranges: Vec<(f64, f64)> = tris
        .iter()
        .map(|t| {
            let h = heights(mesh, t, d);
            (h[0].min(h[1]).min(h[2]), h[0].max(h[1]).max(h[2]))
        })
        .collect();
    // Floor levels, merged within the tolerance; the part's extremes stay exact.
    let mut raw: Vec<f64> = tris.iter().zip(&roles).zip(&ranges).filter(|((_, r), _)| **r == Role::Floor).map(|(_, (a, b))| (a + b) * 0.5).collect();
    raw.push(lo);
    raw.push(hi);
    raw.sort_by(f64::total_cmp);
    let mut levels: Vec<f64> = Vec::new();
    let mut group: Vec<f64> = Vec::new();
    let flush = |group: &mut Vec<f64>, levels: &mut Vec<f64>| {
        if group.is_empty() {
            return;
        }
        let value = if group.iter().any(|h| *h <= lo) {
            lo
        } else if group.iter().any(|h| *h >= hi) {
            hi
        } else {
            group.iter().sum::<f64>() / group.len() as f64
        };
        levels.push(value);
        group.clear();
    };
    for h in raw {
        if group.last().is_some_and(|g| h - g > tolerance) {
            flush(&mut group, &mut levels);
        }
        group.push(h);
    }
    flush(&mut group, &mut levels);
    if levels.len() > MAX_LEVELS {
        return Err(invalid(&format!("too many depth levels along one direction ({}; at most {MAX_LEVELS})", levels.len())));
    }
    let vertex_heights: Vec<f64> = {
        let mut v: Vec<f64> = mesh.positions.iter().map(|p| p.dot(d)).collect();
        v.sort_by(f64::total_cmp);
        v
    };
    let btol = (snap * 0.5).clamp(1e-9, 1.);
    // (start, end, shadow) for each run of slabs with the same shadow.
    let mut runs: Vec<(f64, f64, Vec<Region2>)> = Vec::new();
    for pair in levels.windows(2) {
        let [l0, l1] = pair else { continue };
        let (l0, l1) = (*l0, *l1);
        let varying = tris.iter().zip(&roles).zip(&ranges).any(|((_, r), (a, b))| *r != Role::Wall && *a < l1 - snap && *b > l0 + snap);
        let samples: Vec<f64> = if varying {
            let delta = (snap * 2.).min((l1 - l0) * 0.25);
            let mut inner: Vec<f64> = Vec::new();
            let from = vertex_heights.partition_point(|h| *h <= l0 + delta);
            for h in vertex_heights.iter().skip(from).take_while(|h| **h < l1 - delta) {
                if inner.last().is_none_or(|last| h - last > snap) {
                    inner.push(*h);
                }
            }
            let keep = MAX_SAMPLES - 2;
            let step = inner.len().div_ceil(keep).max(1);
            let mut s = vec![l0 + delta, l1 - delta];
            s.extend(inner.iter().step_by(step));
            s
        } else {
            vec![(l0 + l1) * 0.5]
        };
        let mut shadow: Vec<Region2> = Vec::new();
        for h in samples {
            let regions = section(mesh, tris, &ranges, &plane, d, h, epsilon, snap * 0.2)?;
            shadow = if shadow.is_empty() { regions } else { planar(&shadow, &regions, BoolOp::Union, btol)? };
        }
        if shadow.is_empty() {
            continue;
        }
        if let Some(last) = runs.last_mut()
            && (last.1 - l0).abs() <= tolerance
            && same_shadow(&last.2, &shadow, snap, btol)?
        {
            last.1 = l1;
            continue;
        }
        runs.push((l0, l1, shadow));
    }
    let fitted: Vec<(f64, f64, Vec<Region2>)> = {
        let mut shared = Shared::default();
        runs.into_iter()
            .map(|(a, b, shadow)| Ok((a, b, fit_regions(&shadow, tolerance, snap, epsilon, size, &mut shared)?)))
            .collect::<Result<_>>()?
    };
    // The stack is built by cutting, the way the part was machined: the outline of every
    // section, extruded through, less each pocket open to the top (from its floor up) and to
    // the bottom. Removals open to neither side are left for the other directions.
    let mut outline: Vec<Region2> = Vec::new();
    for (_, _, regions) in &fitted {
        outline = if outline.is_empty() { regions.clone() } else { planar(&outline, regions, BoolOp::Union, btol)? };
    }
    if outline.is_empty() {
        return Err(invalid("no section along a machining direction"));
    }
    let margin = (size * 0.01).max(1.);
    let removed: Vec<Vec<Region2>> = fitted.iter().map(|(_, _, regions)| planar(&outline, regions, BoolOp::Cut, btol)).collect::<Result<_>>()?;
    let mut tools: Vec<(f64, f64, Vec<Region2>)> = Vec::new();
    for from_top in [true, false] {
        // Material removed in every run from this one to the end: a pocket open to that end.
        let order: Vec<usize> = if from_top { (0..fitted.len()).rev().collect() } else { (0..fitted.len()).collect() };
        // (floor, open regions) from the open end inwards; each is inside the one before.
        let mut levels: Vec<(f64, Vec<Region2>)> = Vec::new();
        for i in order {
            let (Some((start, end, _)), Some(here)) = (fitted.get(i), removed.get(i)) else { continue };
            let now = match levels.last() {
                None => here.clone(),
                Some((_, o)) => planar(o, here, BoolOp::Intersect, btol)?,
            };
            let now: Vec<Region2> = now.into_iter().filter(|r| !is_sliver(r, tolerance)).collect();
            if now.is_empty() {
                break;
            }
            levels.push((if from_top { *start } else { *end }, now));
        }
        // A pocket that continues deeper unchanged is cut once, by its deepest tool.
        for (k, (floor, regions)) in levels.iter().enumerate() {
            let deeper = levels.get(k + 1).map(|(_, r)| r.as_slice()).unwrap_or(&[]);
            let mut fresh = Vec::new();
            for r in regions {
                let left: f64 = planar(std::slice::from_ref(r), deeper, BoolOp::Cut, btol)?.iter().map(Region2::area).sum();
                if left > r.area() * 1e-6 + btol {
                    fresh.push(r.clone());
                }
            }
            if !fresh.is_empty() {
                let (a, b) = if from_top { (*floor, hi + margin) } else { (lo - margin, *floor) };
                tools.push((a, b, fresh));
            }
        }
    }
    // Deepest pockets first: a wider, shallower one then contains them instead of meeting
    // their walls face to face.
    tools.sort_by(|a, b| (b.1 - b.0).total_cmp(&(a.1 - a.0)));
    Ok(Stack { plane, lo, hi, margin, outline, tools })
}

/// The region with every loop closed exactly (planar booleans join segments within their
/// tolerance; extrusion checks closure far tighter).
fn sealed(r: &Region2) -> Region2 {
    let seal = |l: &Loop2| Loop2 { segs: close_gaps(l.segs.clone()) };
    Region2 { outer: seal(&r.outer), holes: r.holes.iter().map(seal).collect() }
}

/// A region thinner than the tolerance on average: a fitting sliver, not a pocket.
fn is_sliver(r: &Region2, tolerance: f64) -> bool {
    let perimeter: f64 = std::iter::once(&r.outer).chain(&r.holes).flat_map(|l| &l.segs).map(Seg2::length).sum();
    r.area() <= perimeter * tolerance * 0.5
}

/// A planar boolean without the degenerate loops (zero-area slivers) arrangements of nearly
/// coincident boundaries can leave.
fn planar(a: &[Region2], b: &[Region2], op: BoolOp, tolerance: f64) -> Result<Vec<Region2>> {
    let degenerate = |l: &Loop2| {
        let perimeter: f64 = l.segs.iter().map(Seg2::length).sum();
        l.segs.len() < 2 || l.signed_area().abs() <= perimeter * tolerance
    };
    Ok(crate::planar_regions_boolean(a, b, op, tolerance)?
        .into_iter()
        .filter(|r| !degenerate(&r.outer))
        .map(|mut r| {
            r.holes.retain(|h| !degenerate(h));
            r
        })
        .collect())
}

/// Do two shadows agree within `snap` (their difference is a sliver)?
fn same_shadow(a: &[Region2], b: &[Region2], snap: f64, btol: f64) -> Result<bool> {
    let perimeter: f64 = a.iter().chain(b).flat_map(|r| std::iter::once(&r.outer).chain(&r.holes)).flat_map(|l| &l.segs).map(Seg2::length).sum();
    let limit = perimeter * snap + btol;
    let ab: f64 = planar(a, b, BoolOp::Cut, btol)?.iter().map(Region2::area).sum();
    if ab > limit {
        return Ok(false);
    }
    let ba: f64 = planar(b, a, BoolOp::Cut, btol)?.iter().map(Region2::area).sum();
    Ok(ba <= limit)
}

/// The part's section at height `h` along `d`, in `plane`'s coordinates: outer loops
/// counter-clockwise, holes clockwise.
fn section(
    mesh: &TriMesh,
    tris: &[Tri],
    ranges: &[(f64, f64)],
    plane: &Plane,
    d: Vec3,
    h: f64,
    epsilon: f64,
    collinear: f64,
) -> Result<Vec<Region2>> {
    let above = |i: u32| point(mesh, i).dot(d) > h;
    let cross = |a: u32, b: u32| -> Vec2 {
        // Interpolate from the lower index so both triangles at an edge get the same point.
        let (a, b) = (a.min(b), a.max(b));
        let (pa, pb) = (point(mesh, a), point(mesh, b));
        let (ha, hb) = (pa.dot(d), pb.dot(d));
        let t = if (hb - ha).abs() > 0. { ((h - ha) / (hb - ha)).clamp(0., 1.) } else { 0.5 };
        plane.to_local(pa + (pb - pa) * t)
    };
    // Each segment runs from the side the boundary leaves the slab above (in the triangle's
    // winding) to the side it enters it: material on its left seen from +d.
    let mut next = BTreeMap::<(u32, u32), ((u32, u32), Vec2)>::new();
    for (t, (lo, hi)) in tris.iter().zip(ranges) {
        if *hi <= h || *lo > h {
            continue;
        }
        let [a, b, c] = t.v;
        let mut down = None;
        let mut up = None;
        for (p, q) in [(a, b), (b, c), (c, a)] {
            match (above(p), above(q)) {
                (true, false) => down = Some((p.min(q), p.max(q))),
                (false, true) => up = Some((p.min(q), p.max(q))),
                _ => {}
            }
        }
        let (Some(from), Some(to)) = (down, up) else { continue };
        if next.insert(from, (to, cross(from.0, from.1))).is_some() {
            return Err(invalid("the mesh crosses itself"));
        }
        if next.len() > MAX_SECTION_POINTS {
            return Err(invalid("a section exceeds its geometry budget"));
        }
    }
    let mut loops: Vec<Vec<Vec2>> = Vec::new();
    while let Some((&start, _)) = next.first_key_value() {
        let mut current = start;
        let mut points = Vec::new();
        loop {
            let Some((to, p)) = next.remove(&current) else { return Err(invalid("a section is not closed")) };
            if points.last().is_none_or(|q: &Vec2| q.dist(p) > epsilon) {
                points.push(p);
            }
            current = to;
            if current == start {
                break;
            }
        }
        while points.len() > 1 && points.first().zip(points.last()).is_some_and(|(a, b)| a.dist(*b) <= epsilon) {
            points.pop();
        }
        // Faces crossed by the section leave runs of collinear points (one per triangle).
        let points = drop_collinear(&points, collinear);
        if points.len() >= 3 {
            loops.push(points);
        }
    }
    let loops: Vec<Loop2> = loops.iter().map(|p| Loop2::polygon(p)).filter(|l| l.signed_area().abs() > epsilon * epsilon).collect();
    let mut regions: Vec<Region2> = loops.iter().filter(|l| l.signed_area() > 0.).map(|l| Region2 { outer: l.clone(), holes: Vec::new() }).collect();
    for hole in loops.iter().filter(|l| l.signed_area() < 0.) {
        let Some(probe) = hole.segs.first().map(Seg2::start) else { continue };
        let owner = regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.outer.contains(probe))
            .min_by(|(_, a), (_, b)| a.outer.signed_area().total_cmp(&b.outer.signed_area()))
            .map(|(i, _)| i);
        let Some(region) = owner.and_then(|i| regions.get_mut(i)) else { return Err(invalid("a section hole lies outside the material")) };
        region.holes.push(hole.clone());
    }
    Ok(regions)
}

/// The corners of a polygonal loop (its line starts; other segments sampled).
fn loop_points(lp: &Loop2, tol: f64) -> Vec<Vec2> {
    let mut out = Vec::new();
    for s in &lp.segs {
        match s {
            Seg2::Line { a, .. } => out.push(*a),
            other => {
                let pts = other.polyline(tol);
                out.extend(pts.iter().take(pts.len().saturating_sub(1)));
            }
        }
    }
    out
}

/// Shadow regions with their boundaries fitted as lines and arcs, certified not to cross.
/// Corners and circles already fitted along one direction. The same wall seen in another slab
/// snaps to them, so the slabs' outlines agree exactly instead of within the noise of the mesh.
#[derive(Default)]
struct Shared {
    points: Vec<Vec2>,
    circles: Vec<(Vec2, f64)>,
}

impl Shared {
    fn point(&mut self, p: Vec2, within: f64) -> Vec2 {
        if let Some(q) = self.points.iter().find(|q| q.dist(p) <= within) {
            return *q;
        }
        if self.points.len() < 100_000 {
            self.points.push(p);
        }
        p
    }

    fn circle(&mut self, c: Vec2, r: f64, within: f64) -> (Vec2, f64) {
        if let Some(k) = self.circles.iter().find(|(q, s)| q.dist(c) <= within && (s - r).abs() <= within) {
            return *k;
        }
        if self.circles.len() < 100_000 {
            self.circles.push((c, r));
        }
        (c, r)
    }

    fn snap(&mut self, lp: Loop2, within: f64) -> Loop2 {
        let segs = lp
            .segs
            .into_iter()
            .map(|s| match s {
                Seg2::Line { a, b } => Seg2::Line { a: self.point(a, within), b: self.point(b, within) },
                Seg2::Arc { center, radius, start, sweep } => {
                    let (center, radius) = self.circle(center, radius, within);
                    Seg2::Arc { center, radius, start, sweep }
                }
                other => other,
            })
            .collect();
        Loop2 { segs: close_gaps(segs) }
    }
}

fn fit_regions(shadow: &[Region2], tolerance: f64, snap: f64, epsilon: f64, size: f64, shared: &mut Shared) -> Result<Vec<Region2>> {
    let mut raw = Vec::new();
    let mut fitted = Vec::new();
    let mut out = Vec::new();
    for region in shadow {
        let outer_raw = Loop2::polygon(&loop_points(&region.outer, snap));
        let outer = shared.snap(fit_profile(&loop_points(&region.outer, snap), tolerance, snap), snap);
        let mut holes = Vec::new();
        for hole in &region.holes {
            let points = loop_points(hole, snap);
            let fit = shared.snap(fit_profile(&points, tolerance, snap), snap);
            raw.push(Loop2::polygon(&points));
            fitted.push(fit.clone());
            holes.push(fit.ccw().reversed());
        }
        raw.push(outer_raw);
        fitted.push(outer.clone());
        out.push(Region2 { outer: outer.ccw(), holes });
    }
    crate::mesh_convert::certify_nesting(&raw, &fitted, epsilon, size)?;
    Ok(out)
}

/// The circle through three points: centre and radius.
fn circle3(a: Vec2, b: Vec2, c: Vec2) -> Option<(Vec2, f64)> {
    let u = b - a;
    let v = c - a;
    let d = 2. * u.cross(v);
    if d.abs() <= 1e-18 {
        return None;
    }
    let center = a + Vec2::new(v.y * u.len2() - u.y * v.len2(), u.x * v.len2() - v.x * u.len2()) / d;
    let radius = center.dist(a);
    (radius.is_finite() && center.x.is_finite() && center.y.is_finite()).then_some((center, radius))
}

/// A closed polyline as lines and circular arcs. Arcs are taken only over runs of at least
/// three steps of at most 30° whose vertices lie within `fit` of one circle and whose chords
/// stay within `tolerance` of it; everything else stays straight.
pub(crate) fn fit_profile(points: &[Vec2], tolerance: f64, fit: f64) -> Loop2 {
    let points = &drop_collinear(points, fit * 0.5);
    let n = points.len();
    if !(3..=MAX_FIT_POINTS).contains(&n) {
        return Loop2::polygon(points);
    }
    let get = |i: usize| points.get(i % n).copied().unwrap_or(Vec2::new(0., 0.));
    let turn = |i: usize| {
        let (a, b, c) = (get(i + n - 1), get(i), get(i + 1));
        let (u, v) = (b - a, c - b);
        u.cross(v).atan2(u.dot(v)).abs()
    };
    // Start at the sharpest corner so no straight run or arc is split by the loop's seam; in a
    // loop without corners (tangent lines and arcs), at the start of its longest straight side.
    let sharpest = (0..n).max_by(|a, b| turn(*a).total_cmp(&turn(*b))).unwrap_or(0);
    let start = if turn(sharpest) >= 20f64.to_radians() {
        sharpest
    } else {
        (0..n).max_by(|a, b| get(*a).dist(get(*a + 1)).total_cmp(&get(*b).dist(get(*b + 1)))).unwrap_or(0)
    };
    let at = |i: usize| get(start + i);
    let line_fits = |i: usize, j: usize| {
        let (a, b) = (at(i), at(j));
        if a.dist(b) <= fit {
            return false;
        }
        let edge = Seg2::Line { a, b };
        (i + 1..j).all(|k| {
            let p = at(k);
            edge.point_at(edge.closest_param(p)).dist(p) <= fit
        })
    };
    // (centre, radius, start angle, sweep) of an arc through points i..=j.
    let arc_fits = |i: usize, j: usize| -> Option<(Vec2, f64, f64, f64)> {
        if j < i + 3 {
            return None;
        }
        // Three points spread over the run (its ends coincide when it closes the loop).
        let third = (j - i) / 3;
        let (center, radius) = circle3(at(i), at(i + third), at(i + 2 * third))?;
        if radius <= fit * 10. {
            return None;
        }
        let mut sweep = 0.;
        let mut direction = 0f64;
        for k in i..j {
            let (a, b) = (at(k) - center, at(k + 1) - center);
            if (at(k + 1).dist(center) - radius).abs() > fit {
                return None;
            }
            let step = a.cross(b).atan2(a.dot(b));
            if direction == 0. {
                direction = step.signum();
            }
            if step * direction <= 0. || step.abs() > std::f64::consts::PI / 6. + 1e-9 || radius * (1. - (step.abs() * 0.5).cos()) > tolerance {
                return None;
            }
            sweep += step;
        }
        let full = (sweep.abs() - std::f64::consts::TAU).abs() <= 1e-6;
        if sweep.abs() < 3f64.to_radians() || (sweep.abs() > std::f64::consts::TAU - 1e-6 && !full) {
            return None;
        }
        let a = at(i) - center;
        Some((center, radius, a.y.atan2(a.x), sweep))
    };
    let mut segs: Vec<Seg2> = Vec::new();
    let mut i = 0;
    while i < n {
        let mut line_end = i + 1;
        while line_end < n && line_fits(i, line_end + 1) {
            line_end += 1;
        }
        let mut arc = None;
        let mut j = i + 3;
        while j <= n {
            match arc_fits(i, j) {
                Some(a) => arc = Some((j, a)),
                None => break,
            }
            j += 1;
        }
        match arc {
            Some((j, (center, radius, start, sweep))) if j > line_end => {
                if i == 0 && j == n && (sweep.abs() - std::f64::consts::TAU).abs() <= 1e-6 {
                    let circle = Loop2::circle_from(center, radius, start);
                    return if sweep < 0. { circle.reversed() } else { circle };
                }
                segs.push(Seg2::Arc { center, radius, start, sweep });
                i = j;
            }
            _ => {
                segs.push(Seg2::Line { a: at(i), b: at(line_end) });
                i = line_end;
            }
        }
    }
    Loop2 { segs: close_gaps(merge_segments(segs, fit)) }
}

/// Make each segment start exactly where the one before ends: lines take the end points of
/// the segments beside them (an arc's end, from its angles, sits within the fit of the vertex it
/// was fitted to), and two arcs that meet get a joining line when they miss by more than
/// rounding.
fn close_gaps(segs: Vec<Seg2>) -> Vec<Seg2> {
    let n = segs.len();
    if n < 2 {
        return segs;
    }
    let mut out: Vec<Seg2> = segs.clone();
    for k in 0..n {
        let (Some(prev), Some(next)) = (segs.get((k + n - 1) % n), segs.get((k + 1) % n)) else { continue };
        if let Some(Seg2::Line { a, b }) = out.get_mut(k) {
            *a = prev.end();
            if matches!(next, Seg2::Arc { .. }) {
                *b = next.start();
            }
        }
    }
    let mut joined = Vec::with_capacity(out.len());
    for k in 0..n {
        let (Some(cur), Some(next)) = (out.get(k), out.get((k + 1) % n)) else { continue };
        joined.push(*cur);
        if matches!(cur, Seg2::Arc { .. }) && matches!(next, Seg2::Arc { .. }) && cur.end().dist(next.start()) > 1e-9 {
            joined.push(Seg2::Line { a: cur.end(), b: next.start() });
        }
    }
    joined.retain(|s| !matches!(s, Seg2::Line { a, b } if a.dist(*b) <= 1e-12));
    joined
}

/// The polyline without points lying on the straight line between their neighbours (the
/// section of a faceted wall crosses each facet's diagonal mid-chord).
pub(crate) fn drop_collinear(points: &[Vec2], within: f64) -> Vec<Vec2> {
    let mut out: Vec<Vec2> = points.to_vec();
    if out.len() > MAX_FIT_POINTS * 4 {
        return out;
    }
    let mut changed = true;
    while changed && out.len() > 3 {
        changed = false;
        let mut kept: Vec<Vec2> = Vec::with_capacity(out.len());
        let n = out.len();
        let mut skip_next = false;
        for i in 0..n {
            let (Some(a), Some(b), Some(c)) = (out.get((i + n - 1) % n), out.get(i), out.get((i + 1) % n)) else { continue };
            if !skip_next && kept.len() + (n - i) > 3 {
                let edge = Seg2::Line { a: *a, b: *c };
                let t = edge.closest_param(*b);
                if a.dist(*c) > 0. && t > 0. && t < 1. && edge.point_at(t).dist(*b) <= within {
                    // Drop it; keep its neighbour so two adjacent points are never judged
                    // against each other's removal in one pass.
                    skip_next = true;
                    changed = true;
                    continue;
                }
            }
            skip_next = false;
            kept.push(*b);
        }
        out = kept;
    }
    out
}

/// Join neighbouring arcs of one circle (including across the loop's seam).
fn merge_segments(mut segs: Vec<Seg2>, fit: f64) -> Vec<Seg2> {
    let joined = |a: &Seg2, b: &Seg2| -> Option<Seg2> {
        match (a, b) {
            (Seg2::Arc { center: c0, radius: r0, start, sweep: s0 }, Seg2::Arc { center: c1, radius: r1, sweep: s1, .. })
                if c0.dist(*c1) <= fit && (r0 - r1).abs() <= fit && s0 * s1 > 0. && (s0 + s1).abs() < std::f64::consts::TAU - 1e-6 =>
            {
                Some(Seg2::Arc { center: *c0, radius: *r0, start: *start, sweep: s0 + s1 })
            }
            _ => None,
        }
    };
    let mut changed = true;
    while changed && segs.len() > 2 {
        changed = false;
        let n = segs.len();
        for k in 0..n {
            let (Some(a), Some(b)) = (segs.get(k), segs.get((k + 1) % n)) else { continue };
            if let Some(m) = joined(a, b) {
                if k + 1 < n {
                    segs.splice(k..k + 2, [m]);
                } else {
                    segs.pop();
                    if let Some(first) = segs.first_mut() {
                        *first = m;
                    }
                }
                changed = true;
                break;
            }
        }
    }
    segs
}

/// The converted solid must match the mesh: sound, of the same volume, and on the same side
/// of every triangle (probed just inside and just outside it).
fn certify(result: &Body, mesh: &TriMesh, tris: &[Tri], tolerance: f64, total: f64, epsilon: f64, size: f64) -> Result<()> {
    if !result.validity().is_empty() {
        return Err(invalid("reconstructed solid did not pass validity checks"));
    }
    let source = mesh.to_mesh();
    let original = source.measure().volume;
    let rebuilt = crate::measure(result)?.volume;
    if !original.is_finite() || original <= 0. || (rebuilt - original).abs() > total * tolerance + epsilon * size * size {
        return Err(invalid("reconstructed volume is inconsistent with the source mesh"));
    }
    let converted = result.tessellate(tolerance * 0.25)?;
    let inside_result = converted.inside_index();
    let inside_source = source.inside_index();
    let mut wrong = 0.;
    for t in tris {
        let [a, b, c] = t.v;
        let centroid = (point(mesh, a) + point(mesh, b) + point(mesh, c)) / 3.;
        let disagrees = |k: f64| -> Option<bool> {
            let (inner, outer) = (centroid - t.n * k, centroid + t.n * k);
            if !inside_source.contains(inner) || inside_source.contains(outer) {
                return None;
            }
            Some(!inside_result.contains(inner) || inside_result.contains(outer))
        };
        if disagrees(tolerance * 2.) == Some(true) && disagrees(tolerance * 4.).unwrap_or(true) {
            wrong += t.area;
        }
    }
    if wrong > total * 1e-4 {
        return Err(invalid(&format!(
            "{:.2}% of the surface could not be reproduced within the tolerance (undercuts or features not cut along a machining direction)",
            wrong / total * 100.
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh_of(body: &Body) -> Body {
        let mesh = body.tessellate(0.005).unwrap();
        crate::mesh_body(&mesh.positions, &mesh.triangles).unwrap()
    }

    fn cut(a: &Body, b: &Body) -> Body {
        crate::boolean(a, b, BoolOp::Cut).unwrap().unwrap()
    }

    fn volume(b: &Body) -> f64 {
        crate::measure(b).unwrap().volume
    }

    #[test]
    fn fits_lines_arcs_and_full_circles() {
        let circle: Vec<_> = (0..48)
            .map(|i| Vec2::new(5., 0.) + Vec2::new((i as f64 * 7.5f64.to_radians()).cos(), (i as f64 * 7.5f64.to_radians()).sin()) * 3.)
            .collect();
        let fitted = fit_profile(&circle, 0.05, 1e-6);
        assert_eq!(fitted.segs.len(), 2);
        assert!((fitted.signed_area() - std::f64::consts::PI * 9.).abs() < 1e-6);
        // A 20 x 10 rectangle with corners rounded at radius 2, tessellated at 10°.
        let mut pts = Vec::new();
        for (cx, cy, a0) in [(18., 2., -90.), (18., 8., 0.), (2., 8., 90.), (2., 2., 180.)] {
            for k in 0..=9 {
                let a = (a0 + k as f64 * 10f64).to_radians();
                pts.push(Vec2::new(cx + 2. * a.cos(), cy + 2. * a.sin()));
            }
        }
        let fitted = fit_profile(&pts, 0.05, 1e-6);
        let arcs = fitted.segs.iter().filter(|s| matches!(s, Seg2::Arc { .. })).count();
        let lines = fitted.segs.iter().filter(|s| matches!(s, Seg2::Line { .. })).count();
        assert_eq!((arcs, lines), (4, 4), "{:?}", fitted.segs);
        assert!((fitted.signed_area() - (200. - (4. - std::f64::consts::PI) * 4.)).abs() < 1e-6);
        // A square stays four lines.
        let square = [Vec2::new(0., 0.), Vec2::new(5., 0.), Vec2::new(10., 0.), Vec2::new(10., 10.), Vec2::new(0., 10.)];
        assert_eq!(fit_profile(&square, 0.05, 1e-6).segs.len(), 4);
    }

    #[test]
    fn block_with_pocket_and_holes_from_three_sides() {
        let block = crate::box_solid(Vec3::ZERO, Vec3::new(60., 40., 25.)).unwrap();
        // Pocket from the top, a through hole from the top, a blind hole from the front and a
        // through hole from the side.
        let pocket = crate::box_solid(Vec3::new(10., 10., 15.), Vec3::new(30., 30., 26.)).unwrap();
        let top_hole = crate::cylinder(Vec3::new(45., 20., -1.), Vec3::Z, 4., 27.).unwrap();
        let front_hole = crate::cylinder(Vec3::new(45., -1., 8.), Vec3::Y, 3., 16.).unwrap();
        let side_hole = crate::cylinder(Vec3::new(-1., 33., 6.), Vec3::X, 2.5, 62.).unwrap();
        let mut part = block;
        for tool in [&pocket, &top_hole, &front_hole, &side_hole] {
            part = cut(&part, tool);
        }
        let expected = volume(&part);
        let converted = crate::convert_prismatic_mesh(&mesh_of(&part), 0.05).unwrap();
        assert!(!converted.is_mesh());
        assert!(converted.validity().is_empty());
        assert!((volume(&converted) - expected).abs() < 0.5, "{} vs {expected}", volume(&converted));
        let cylinders = crate::measure(&converted).unwrap().merged.face_types.get("cylinder").copied().unwrap_or(0);
        assert!(cylinders >= 3, "{cylinders} cylinders");
    }

    #[test]
    fn stepped_plate_with_rounded_pocket_corners() {
        let base = crate::box_solid(Vec3::ZERO, Vec3::new(50., 30., 10.)).unwrap();
        let step = crate::box_solid(Vec3::new(35., -1., 6.), Vec3::new(51., 31., 11.)).unwrap();
        let region = Region2 {
            outer: Loop2 {
                segs: vec![
                    Seg2::Line { a: Vec2::new(8., 5.), b: Vec2::new(22., 5.) },
                    Seg2::Arc { center: Vec2::new(22., 8.), radius: 3., start: -std::f64::consts::FRAC_PI_2, sweep: std::f64::consts::FRAC_PI_2 },
                    Seg2::Line { a: Vec2::new(25., 8.), b: Vec2::new(25., 22.) },
                    Seg2::Arc { center: Vec2::new(22., 22.), radius: 3., start: 0., sweep: std::f64::consts::FRAC_PI_2 },
                    Seg2::Line { a: Vec2::new(22., 25.), b: Vec2::new(8., 25.) },
                    Seg2::Arc { center: Vec2::new(8., 22.), radius: 3., start: std::f64::consts::FRAC_PI_2, sweep: std::f64::consts::FRAC_PI_2 },
                    Seg2::Line { a: Vec2::new(5., 22.), b: Vec2::new(5., 8.) },
                    Seg2::Arc { center: Vec2::new(8., 8.), radius: 3., start: std::f64::consts::PI, sweep: std::f64::consts::FRAC_PI_2 },
                ],
            },
            holes: Vec::new(),
        };
        let pocket = crate::extrude_planar_regions(&Plane::XY, &[region], 4., 11.).unwrap();
        let part = cut(&cut(&base, &step), &pocket);
        let expected = volume(&part);
        let converted = crate::convert_prismatic_mesh(&mesh_of(&part), 0.05).unwrap();
        assert!((volume(&converted) - expected).abs() < 0.2, "{} vs {expected}", volume(&converted));
        assert_eq!(crate::measure(&converted).unwrap().merged.face_types.get("cylinder"), Some(&4));
    }

    /// The mesh as an STL file stores it: single-precision coordinates.
    fn stl_of(body: &Body) -> Body {
        let mesh = body.tessellate(0.005).unwrap();
        let positions: Vec<Vec3> = mesh.positions.iter().map(|p| Vec3::new(p.x as f32 as f64, p.y as f32 as f64, p.z as f32 as f64)).collect();
        crate::mesh_body(&positions, &mesh.triangles).unwrap()
    }

    #[test]
    fn counterbore_chamfer_and_single_precision_rotated_part() {
        // A 40 x 30 x 20 block, chamfered 4 mm along one edge in X, with a counterbored hole
        // from the top and a slot from the side, then turned off the model axes.
        let block = crate::box_solid(Vec3::ZERO, Vec3::new(40., 30., 20.)).unwrap();
        let wedge = Region2 { outer: Loop2::polygon(&[Vec2::new(26., 20.), Vec2::new(31., 20.), Vec2::new(31., 15.)]), holes: Vec::new() };
        let chamfer = crate::extrude_planar_regions(&Plane::YZ, &[wedge], -1., 41.).unwrap();
        let bore = crate::cylinder(Vec3::new(12., 15., -1.), Vec3::Z, 3., 22.).unwrap();
        let counterbore = crate::cylinder(Vec3::new(12., 15., 14.), Vec3::Z, 5.5, 7.).unwrap();
        let slot = crate::box_solid(Vec3::new(30., 8., 4.), Vec3::new(41., 18., 10.)).unwrap();
        let mut part = block;
        for tool in [&chamfer, &bore, &counterbore, &slot] {
            part = cut(&part, tool);
        }
        let expected = volume(&part);
        for (turned, part) in
            [(false, part.clone()), (true, crate::transform(&part, Vec3::new(5., -3., 2.), Vec3::ZERO, Vec3::new(1., 2., 3.), 0.6).unwrap())]
        {
            let start = std::time::Instant::now();
            let converted = crate::convert_prismatic_mesh(&stl_of(&part), 0.05).unwrap_or_else(|e| panic!("turned {turned}: {e}"));
            assert!(converted.validity().is_empty());
            assert!((volume(&converted) - expected).abs() < 0.5, "turned {turned}: {} vs {expected}", volume(&converted));
            let types = crate::measure(&converted).unwrap().merged.face_types;
            assert_eq!(types.get("cylinder"), Some(&2), "turned {turned}: {types:?}");
            eprintln!("turned {turned}: {:?} in {:?}", types, start.elapsed());
        }
    }

    #[test]
    fn drill_points_and_spheres_are_refused() {
        let block = crate::box_solid(Vec3::ZERO, Vec3::new(20., 20., 20.)).unwrap();
        let ball = crate::sphere(Vec3::new(10., 10., 20.), 6.).unwrap();
        let err = crate::convert_prismatic_mesh(&mesh_of(&cut(&block, &ball)), 0.05).unwrap_err().to_string();
        assert!(err.contains("machining directions"), "{err}");
    }
}
