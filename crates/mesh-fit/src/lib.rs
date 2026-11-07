//! Reverse engineering of mesh bodies (STL, 3MF, OBJ): surfaces recognised where they are
//! clicked, solids, sheets and sketches fitted to them, milled parts rebuilt along their machining
//! directions, sections, alignment and deviation checks.
//!
//! Built only on the kernel's public API: the kernel itself is not changed. The mesh is read
//! through [`Body::tessellate`] and welded back to shared vertices here.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use solidcraft_geom::{Loop2, Region2, Vec3};
use solidcraft_kernel::{Body, KernelError, Result};

mod milled;
mod regions;

pub use regions::{
    Deviation, MeshRegion, Primitive, SurfaceKind, align_to_axes, fit_bodies, mesh_deviation, picks_for_all, recognize_all, recognize_region,
    section_outline, surface_outline,
};

const MAX_TRIANGLES: usize = 50_000;
/// Nesting certification runs two planar booleans per pair of boundary loops, so the loop count
/// bounds its cost.
const MAX_LOOPS: usize = 50;

/// Indexed triangles with shared vertices, wound counter-clockwise seen from outside.
#[derive(Clone, Debug, Default)]
pub(crate) struct TriMesh {
    pub(crate) positions: Vec<Vec3>,
    pub(crate) triangles: Vec<[u32; 3]>,
}

impl TriMesh {
    /// The display mesh: corners split per triangle so normals stay flat.
    pub(crate) fn to_mesh(&self) -> solidcraft_geom::Mesh {
        let mut out = solidcraft_geom::Mesh::default();
        out.positions.reserve(self.triangles.len() * 3);
        for t in &self.triangles {
            let (Some(a), Some(b), Some(c)) =
                (self.positions.get(t[0] as usize), self.positions.get(t[1] as usize), self.positions.get(t[2] as usize))
            else {
                continue;
            };
            let n = (*b - *a).cross(*c - *a).normalized().unwrap_or(Vec3::Z);
            let base = u32::try_from(out.positions.len()).unwrap_or(u32::MAX);
            out.positions.extend([*a, *b, *c]);
            out.normals.extend([n, n, n]);
            out.triangles.push([base, base.saturating_add(1), base.saturating_add(2)]);
            out.tri_face.push(0);
        }
        out
    }
}

fn invalid(message: &str) -> KernelError {
    KernelError::Invalid(format!("mesh conversion: {message}"))
}

fn over_budget(triangles: usize) -> KernelError {
    invalid(&format!(
        "the mesh has {triangles} triangles, more than the {MAX_TRIANGLES} conversion can take; only closed constant-section parts (one profile extruded straight) and milled parts (planes and holes along a few machining directions) can be converted, so a freeform mesh is refused at any size"
    ))
}

/// The triangles of a mesh body, one corner triple per triangle as the kernel hands them out.
pub(crate) fn mesh_of(body: &Body) -> Result<TriMesh> {
    if !body.is_mesh() {
        return Err(invalid("select a mesh body"));
    }
    let m = body.tessellate(0.05)?;
    if m.triangles.len() > MAX_TRIANGLES {
        return Err(over_budget(m.triangles.len()));
    }
    // The kernel hands each triangle its own corners: share the ones at the same point again,
    // as the body holds them.
    let mut index = std::collections::HashMap::<[u64; 3], u32>::new();
    let mut positions = Vec::new();
    let mut remap = Vec::with_capacity(m.positions.len());
    for p in &m.positions {
        let key = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
        let next = u32::try_from(positions.len()).map_err(|_| invalid("too many vertices"))?;
        let i = *index.entry(key).or_insert_with(|| {
            positions.push(*p);
            next
        });
        remap.push(i);
    }
    let triangles = m
        .triangles
        .iter()
        .map(|t| {
            let (Some(&a), Some(&b), Some(&c)) = (remap.get(t[0] as usize), remap.get(t[1] as usize), remap.get(t[2] as usize)) else {
                return Err(invalid("missing triangle vertex"));
            };
            Ok([a, b, c])
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TriMesh { positions, triangles })
}

/// The mesh with its coincident corners merged into shared vertices.
pub(crate) fn weld(mesh: &TriMesh, tolerance: f64) -> Result<TriMesh> {
    if mesh.triangles.len() > MAX_TRIANGLES {
        return Err(over_budget(mesh.triangles.len()));
    }
    if mesh.positions.len() > MAX_TRIANGLES * 3 {
        return Err(invalid(&format!("the mesh has {} vertices, more than the {} conversion can take", mesh.positions.len(), MAX_TRIANGLES * 3)));
    }
    let bounds = mesh.positions.iter().fold(solidcraft_geom::Aabb3::EMPTY, |mut b, p| {
        b.add(*p);
        b
    });
    let epsilon = (bounds.size().len().max(1.) * 1e-7).min(tolerance * 0.01).max(1e-9);
    if !epsilon.is_finite() || mesh.positions.iter().any(|p| [p.x, p.y, p.z].iter().any(|v| !v.is_finite() || (v / epsilon).abs() > 1e15)) {
        return Err(invalid("mesh coordinates exceed the conversion budget"));
    }
    let key = |p: Vec3| ((p.x / epsilon).floor() as i64, (p.y / epsilon).floor() as i64, (p.z / epsilon).floor() as i64);
    let mut grid = BTreeMap::<_, Vec<u32>>::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let mut remap = Vec::new();
    let mut comparisons = 0usize;
    for &p in &mesh.positions {
        let k = key(p);
        let mut found = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &i in grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)).into_iter().flatten() {
                        comparisons += 1;
                        if comparisons > 8_000_000 {
                            return Err(invalid("vertex welding exceeds its work budget"));
                        }
                        if positions.get(i as usize).is_some_and(|q| q.dist(p) <= epsilon) {
                            found = Some(i);
                            break;
                        }
                    }
                }
            }
        }
        let i = match found {
            Some(i) => i,
            None => {
                let i = u32::try_from(positions.len()).map_err(|_| invalid("too many vertices"))?;
                positions.push(p);
                grid.entry(k).or_default().push(i);
                i
            }
        };
        remap.push(i);
    }
    let triangles = mesh
        .triangles
        .iter()
        .map(|t| {
            let (Some(&a), Some(&b), Some(&c)) = (remap.get(t[0] as usize), remap.get(t[1] as usize), remap.get(t[2] as usize)) else {
                return Err(invalid("missing triangle vertex"));
            };
            Ok([a, b, c])
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TriMesh { positions, triangles })
}

/// The welded mesh of a mesh body.
pub(crate) fn welded(body: &Body, tolerance: f64) -> Result<TriMesh> {
    weld(&mesh_of(body)?, tolerance)
}

/// The kernel's constant-section conversion, else a milled part: walls, pockets, steps and holes
/// along up to six machining directions, guided by picked surfaces (each a point on the mesh and
/// the kind of surface there). The mesh's vertices are put onto the picked surfaces first, so
/// their exact sizes carry over and noise does not; picked cones and spheres are cut as solids of
/// their own. The result is certified against the original mesh.
pub fn convert_mesh_with_regions(body: &Body, tolerance: f64, picks: &[(Vec3, SurfaceKind)]) -> Result<Body> {
    solidcraft_kernel::guard("convert a mesh along its machining directions", || {
        if !tolerance.is_finite() || !(1e-6..=1.).contains(&tolerance) {
            return Err(invalid("tolerance must be between 0.000001 and 1 mm"));
        }
        if picks.len() > 10_000 || picks.iter().any(|(p, _)| !p.is_finite()) {
            return Err(invalid("picks must be finite points, at most 10000"));
        }
        let prism = if picks.is_empty() { Some(solidcraft_kernel::convert_prismatic_mesh(body, tolerance)) } else { None };
        if let Some(Ok(solid)) = prism {
            return Ok(solid);
        }
        let mesh = welded(body, tolerance)?;
        let mut edges: BTreeMap<(u32, u32), (usize, i32)> = BTreeMap::new();
        for &[a, b, c] in &mesh.triangles {
            for (a, b) in [(a, b), (b, c), (c, a)] {
                let entry = edges.entry((a.min(b), a.max(b))).or_default();
                entry.0 += 1;
                entry.1 += if a < b { 1 } else { -1 };
            }
        }
        if edges.values().any(|&(count, balance)| count != 2 || balance != 0) {
            return Err(invalid("mesh must be closed and consistently oriented, with two triangles per edge"));
        }
        let (snapped, features) = if picks.is_empty() {
            (None, milled::Features::default())
        } else {
            regions::snap_to_picks(&mesh, tolerance, picks).map(|(m, f)| (Some(m), f))?
        };
        let target = snapped.as_ref().unwrap_or(&mesh);
        milled::convert_milled_with(target, &mesh, tolerance, &features).map_err(|e| {
            let reason = e.to_string();
            let reason = reason.strip_prefix("invalid input: mesh conversion: ").unwrap_or(&reason);
            let prism = match &prism {
                Some(Err(e)) => format!("; as a constant-section prism: {e}"),
                _ => String::new(),
            };
            invalid(&format!("this mesh is neither a constant-section prism nor a milled part ({reason}{prism}); no geometry was converted"))
        })
    })
}

pub(crate) fn certify_nesting(raw: &[Loop2], fitted: &[Loop2], epsilon: f64, size: f64) -> Result<()> {
    if raw.len() > MAX_LOOPS {
        return Err(KernelError::Invalid(format!(
            "mesh conversion: too complex to convert ({} boundary loops; at most {MAX_LOOPS} are supported)",
            raw.len()
        )));
    }
    let mut work = 0usize;
    for (i, a) in raw.iter().enumerate() {
        for (j, b) in raw.iter().enumerate().skip(i + 1) {
            work = work.saturating_add(a.segs.len().saturating_mul(b.segs.len()));
            if work > 16_000_000 {
                return Err(invalid("loop certification exceeds its work budget"));
            }
            let contains = |a: &Loop2, b: &Loop2| b.segs.first().is_some_and(|s| a.contains(s.start()));
            let (ab, ba) = (contains(a, b), contains(b, a));
            if ab && ba {
                return Err(invalid("overlapping boundary loops"));
            }
            for (a, b) in
                [(a, b), (fitted.get(i).ok_or_else(|| invalid("missing fitted loop"))?, fitted.get(j).ok_or_else(|| invalid("missing fitted loop"))?)]
            {
                let region = |l: &Loop2| Region2 { outer: l.ccw(), holes: Vec::new() };
                let overlap = solidcraft_kernel::planar_regions_boolean(
                    &[region(a)],
                    &[region(b)],
                    solidcraft_kernel::BoolOp::Intersect,
                    epsilon.clamp(1e-9, 1.),
                )?;
                let area: f64 = overlap.iter().map(Region2::area).sum();
                let expected = if ab {
                    b.signed_area().abs()
                } else if ba {
                    a.signed_area().abs()
                } else {
                    0.
                };
                if (area - expected).abs() > epsilon * size.max(1.) * 10. {
                    return Err(invalid("conversion would cross or merge boundary loops"));
                }
            }
        }
    }
    Ok(())
}

/// A mesh body from `body`'s tessellation, welded as a mesh export welds it (the kernel's
/// faces meet at nearly, not exactly, the same points).
#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(crate) fn tests_mesh(body: &Body, chord: f64) -> Body {
    let m = body.tessellate(chord).unwrap();
    let welded = weld(&TriMesh { positions: m.positions, triangles: m.triangles }, 1e-4).unwrap();
    solidcraft_kernel::mesh_body(&welded.positions, &welded.triangles).unwrap()
}
