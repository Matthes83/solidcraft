//! Recognising analytic surfaces in a mesh body, the way a part is reverse-engineered by hand:
//! pick a point on the mesh and the connected region around it that one plane, cylinder, cone
//! or sphere fits within the tolerance is found, with that surface's parameters.
//!
//! The region grows from the picked triangle across neighbours that lie on the current fit
//! (every corner within the tolerance, the facet facing the same way as the surface). Picked
//! without a kind, it starts as a plane and moves on to a cylinder, cone or sphere when the
//! curved neighbours fit one of those instead, so a pick on a bore finds the whole bore and a
//! pick on a flat face stops at its edges and at tangent fillets.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use solidcraft_geom::Vec3;

use crate::meshbody::TriMesh;
use crate::{Body, KernelError, Result};

/// Triangles a region (or a whole segmentation) may cover.
const MAX_TRIANGLES: usize = 200_000;
/// Regions one segmentation of a whole mesh returns at most.
const MAX_REGIONS: usize = 2000;
/// Neighbours across a fold sharper than this never start a curved fit.
const SMOOTH_FOLD: f64 = 30.;
/// A facet may tilt this far from the surface's normal at its centre (coarse tessellations).
const FACET_TILT: f64 = 25.;

fn invalid(message: &str) -> KernelError {
    KernelError::Invalid(format!("surface recognition: {message}"))
}

/// The surface to recognise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKind {
    /// Whichever of the others fits, simplest first.
    #[default]
    Auto,
    Plane,
    Cylinder,
    Cone,
    Sphere,
    /// A ring: a fillet or round swept about an axis.
    Torus,
}

impl SurfaceKind {
    pub fn parse(s: &str) -> Option<SurfaceKind> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => SurfaceKind::Auto,
            "plane" => SurfaceKind::Plane,
            "cylinder" => SurfaceKind::Cylinder,
            "cone" => SurfaceKind::Cone,
            "sphere" => SurfaceKind::Sphere,
            "torus" => SurfaceKind::Torus,
            _ => return None,
        })
    }
}

/// An analytic surface fitted to a region. Axes are unit vectors; a cone's axis points from
/// its apex into the cone.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Primitive {
    Plane {
        point: Vec3,
        normal: Vec3,
    },
    Cylinder {
        point: Vec3,
        axis: Vec3,
        radius: f64,
    },
    Cone {
        apex: Vec3,
        axis: Vec3,
        half_angle: f64,
    },
    Sphere {
        center: Vec3,
        radius: f64,
    },
    /// The surface a circle of radius `minor` sweeps, its centre `major` from `center` in the
    /// plane across `axis`.
    Torus {
        center: Vec3,
        axis: Vec3,
        major: f64,
        minor: f64,
    },
}

impl Primitive {
    pub fn kind(&self) -> SurfaceKind {
        match self {
            Primitive::Plane { .. } => SurfaceKind::Plane,
            Primitive::Cylinder { .. } => SurfaceKind::Cylinder,
            Primitive::Cone { .. } => SurfaceKind::Cone,
            Primitive::Sphere { .. } => SurfaceKind::Sphere,
            Primitive::Torus { .. } => SurfaceKind::Torus,
        }
    }

    /// Signed distance from the surface, positive on the side its [`Primitive::normal_at`] points to.
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        match *self {
            Primitive::Plane { point, normal } => (p - point).dot(normal),
            Primitive::Cylinder { point, axis, radius } => {
                let w = p - point;
                (w - axis * w.dot(axis)).len() - radius
            }
            Primitive::Cone { apex, axis, half_angle } => {
                let w = p - apex;
                let h = w.dot(axis);
                let rho = (w - axis * h).len();
                rho * half_angle.cos() - h * half_angle.sin()
            }
            Primitive::Sphere { center, radius } => p.dist(center) - radius,
            Primitive::Torus { center, axis, major, minor } => {
                let w = p - center;
                let h = w.dot(axis);
                let rho = (w - axis * h).len();
                (rho - major).hypot(h) - minor
            }
        }
    }

    /// The surface's unit normal near `p`, pointing away from its axis or centre.
    pub fn normal_at(&self, p: Vec3) -> Vec3 {
        match *self {
            Primitive::Plane { normal, .. } => normal,
            Primitive::Cylinder { point, axis, .. } => {
                let w = p - point;
                (w - axis * w.dot(axis)).normalized().unwrap_or_else(|| axis.any_perp())
            }
            Primitive::Cone { apex, axis, half_angle } => {
                let w = p - apex;
                let radial = (w - axis * w.dot(axis)).normalized().unwrap_or_else(|| axis.any_perp());
                (radial * half_angle.cos() - axis * half_angle.sin()).normalized().unwrap_or(radial)
            }
            Primitive::Sphere { center, .. } => (p - center).normalized().unwrap_or(Vec3::Z),
            Primitive::Torus { center, axis, major, .. } => {
                let w = p - center;
                let radial = (w - axis * w.dot(axis)).normalized().unwrap_or_else(|| axis.any_perp());
                (p - (center + radial * major)).normalized().unwrap_or(radial)
            }
        }
    }

    /// The nearest point on the surface.
    pub fn project(&self, p: Vec3) -> Vec3 {
        p - self.normal_at(p) * self.signed_distance(p)
    }
}

/// A recognised region of a mesh body.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MeshRegion {
    pub surface: Primitive,
    /// The body's triangles (indices into its mesh) the surface covers.
    pub triangles: Vec<u32>,
    pub area: f64,
    /// Largest and root-mean-square distance of the region's vertices from the surface.
    pub max_deviation: f64,
    pub rms_deviation: f64,
    /// The mesh lies on the surface's axis side: a bore or a spherical pocket, not a boss or a
    /// ball. Always false for a plane, whose normal points out of the material.
    pub concave: bool,
}

/// A welded mesh with what region growing asks of it.
pub(crate) struct Ctx {
    pub(crate) positions: Vec<Vec3>,
    pub(crate) triangles: Vec<[u32; 3]>,
    pub(crate) normals: Vec<Vec3>,
    pub(crate) areas: Vec<f64>,
    pub(crate) neighbours: Vec<Vec<u32>>,
}

impl Ctx {
    pub(crate) fn new(mesh: &TriMesh) -> Result<Ctx> {
        if mesh.triangles.len() > MAX_TRIANGLES {
            return Err(invalid(&format!("the mesh has {} triangles, more than the {MAX_TRIANGLES} recognition can take", mesh.triangles.len())));
        }
        let mut normals = Vec::with_capacity(mesh.triangles.len());
        let mut areas = Vec::with_capacity(mesh.triangles.len());
        let mut sides = std::collections::BTreeMap::<(u32, u32), Vec<u32>>::new();
        for (i, t) in mesh.triangles.iter().enumerate() {
            let [a, b, c] = *t;
            let (Some(pa), Some(pb), Some(pc)) = (mesh.positions.get(a as usize), mesh.positions.get(b as usize), mesh.positions.get(c as usize))
            else {
                return Err(invalid("triangle refers to a missing vertex"));
            };
            let cross = (*pb - *pa).cross(*pc - *pa);
            normals.push(cross.normalized().unwrap_or(Vec3::ZERO));
            areas.push(cross.len() * 0.5);
            let i = u32::try_from(i).map_err(|_| invalid("too many triangles"))?;
            for (p, q) in [(a, b), (b, c), (c, a)] {
                sides.entry((p.min(q), p.max(q))).or_default().push(i);
            }
        }
        let mut neighbours = vec![Vec::new(); mesh.triangles.len()];
        for tris in sides.values() {
            for &t in tris {
                for &u in tris {
                    if t != u
                        && let Some(n) = neighbours.get_mut(t as usize)
                    {
                        n.push(u);
                    }
                }
            }
        }
        Ok(Ctx { positions: mesh.positions.clone(), triangles: mesh.triangles.clone(), normals, areas, neighbours })
    }

    pub(crate) fn corners(&self, t: u32) -> [Vec3; 3] {
        let [a, b, c] = self.triangles.get(t as usize).copied().unwrap_or([0, 0, 0]);
        let p = |i: u32| self.positions.get(i as usize).copied().unwrap_or(Vec3::ZERO);
        [p(a), p(b), p(c)]
    }

    fn normal(&self, t: u32) -> Vec3 {
        self.normals.get(t as usize).copied().unwrap_or(Vec3::ZERO)
    }

    fn area(&self, t: u32) -> f64 {
        self.areas.get(t as usize).copied().unwrap_or(0.)
    }

    fn centroid(&self, t: u32) -> Vec3 {
        let [a, b, c] = self.corners(t);
        (a + b + c) / 3.
    }

    fn neighbours(&self, t: u32) -> &[u32] {
        self.neighbours.get(t as usize).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Does triangle `t` lie on `s`: corners within `tol` and facing along its normal?
    fn on(&self, t: u32, s: &Primitive, tol: f64) -> bool {
        let n = self.normal(t);
        // A sliver's normal is noise: only its corners count.
        let sliver = self.area(t) <= tol * tol * 4.;
        self.corners(t).iter().all(|p| s.signed_distance(*p).abs() <= tol)
            && (sliver || (n.len() > 0.5 && s.normal_at(self.centroid(t)).dot(n).abs() >= FACET_TILT.to_radians().cos()))
    }

    /// The triangle nearest `p`.
    pub(crate) fn nearest(&self, p: Vec3) -> Option<u32> {
        let mut best: Option<(u32, f64)> = None;
        for t in 0..self.triangles.len() {
            let t = u32::try_from(t).ok()?;
            if self.area(t) <= 0. {
                continue;
            }
            let [a, b, c] = self.corners(t);
            let d = closest_on_triangle(p, a, b, c).dist(p);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((t, d));
            }
        }
        best.map(|(t, _)| t)
    }
}

/// The point of triangle `abc` nearest `p`.
fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0. && d2 <= 0. {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0. && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0. && d1 >= 0. && d3 <= 0. {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0. && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0. && d2 >= 0. && d6 <= 0. {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0. && (d4 - d3) >= 0. && (d5 - d6) >= 0. {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = va + vb + vc;
    if denom.abs() <= f64::MIN_POSITIVE {
        return a;
    }
    a + ab * (vb / denom) + ac * (vc / denom)
}

type M3 = [[f64; 3]; 3];

fn outer(m: &mut M3, v: Vec3, w: f64) {
    let a = [v.x, v.y, v.z];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell += w * a.get(i).copied().unwrap_or(0.) * a.get(j).copied().unwrap_or(0.);
        }
    }
}

/// Eigenvalues (ascending) and unit eigenvectors of a symmetric 3x3 matrix (Jacobi rotations).
fn eigen(m: M3) -> [(f64, Vec3); 3] {
    let mut a = m;
    let mut v: M3 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let get = |m: &M3, i: usize, j: usize| m.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.);
    let set = |m: &mut M3, i: usize, j: usize, x: f64| {
        if let Some(c) = m.get_mut(i).and_then(|r| r.get_mut(j)) {
            *c = x;
        }
    };
    for _ in 0..50 {
        let off = get(&a, 0, 1).abs() + get(&a, 0, 2).abs() + get(&a, 1, 2).abs();
        if off < 1e-15 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = get(&a, p, q);
            if apq.abs() < 1e-18 {
                continue;
            }
            let theta = (get(&a, q, q) - get(&a, p, p)) / (2. * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.).sqrt());
            let t = if theta == 0. { 1. } else { t };
            let c = 1. / (t * t + 1.).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (akp, akq) = (get(&a, k, p), get(&a, k, q));
                set(&mut a, k, p, c * akp - s * akq);
                set(&mut a, k, q, s * akp + c * akq);
            }
            for k in 0..3 {
                let (apk, aqk) = (get(&a, p, k), get(&a, q, k));
                set(&mut a, p, k, c * apk - s * aqk);
                set(&mut a, q, k, s * apk + c * aqk);
            }
            for k in 0..3 {
                let (vkp, vkq) = (get(&v, k, p), get(&v, k, q));
                set(&mut v, k, p, c * vkp - s * vkq);
                set(&mut v, k, q, s * vkp + c * vkq);
            }
        }
    }
    let mut out = [0, 1, 2].map(|i| (get(&a, i, i), Vec3::new(get(&v, 0, i), get(&v, 1, i), get(&v, 2, i)).normalized().unwrap_or(Vec3::Z)));
    out.sort_by(|x, y| x.0.total_cmp(&y.0));
    out
}

/// Solve the `n`x`n` system `a x = b` (Gaussian elimination with pivoting).
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let size = |i: usize| a.get(i).and_then(|r| r.get(col)).map_or(0., |x| x.abs());
        let pivot = (col..N).max_by(|&i, &j| size(i).total_cmp(&size(j)))?;
        a.swap(col, pivot);
        b.swap(col, pivot);
        let p = *a.get(col)?.get(col)?;
        if p.abs() < 1e-14 {
            return None;
        }
        for row in 0..N {
            if row == col {
                continue;
            }
            let f = a.get(row)?.get(col)? / p;
            if f == 0. {
                continue;
            }
            let pivot_row = *a.get(col)?;
            let target = a.get_mut(row)?;
            for (x, y) in target.iter_mut().zip(pivot_row.iter()) {
                *x -= f * y;
            }
            let bc = *b.get(col)?;
            *b.get_mut(row)? -= f * bc;
        }
    }
    let mut x = [0.; N];
    for i in 0..N {
        *x.get_mut(i)? = b.get(i)? / a.get(i)?.get(i)?;
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

fn vertices(ctx: &Ctx, tris: &[u32]) -> Vec<Vec3> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for &t in tris {
        for i in ctx.triangles.get(t as usize).copied().unwrap_or([0, 0, 0]) {
            if seen.insert(i)
                && let Some(p) = ctx.positions.get(i as usize)
            {
                out.push(*p);
            }
        }
    }
    out
}

fn fit_plane(ctx: &Ctx, tris: &[u32]) -> Option<Primitive> {
    let pts = vertices(ctx, tris);
    if pts.len() < 3 {
        return None;
    }
    let c = pts.iter().fold(Vec3::ZERO, |s, p| s + *p) / pts.len() as f64;
    let mut m: M3 = [[0.; 3]; 3];
    for p in &pts {
        outer(&mut m, *p - c, 1.);
    }
    let normal = eigen(m).first()?.1;
    let facing = tris.iter().fold(Vec3::ZERO, |s, t| s + ctx.normal(*t) * ctx.area(*t));
    let normal = if facing.dot(normal) < 0. { -normal } else { normal };
    Some(Primitive::Plane { point: c, normal })
}

/// A cylinder's axis: the lines where neighbouring facets meet run along it.
fn ruling_axis(ctx: &Ctx, tris: &[u32]) -> Option<Vec3> {
    let inside: std::collections::BTreeSet<u32> = tris.iter().copied().collect();
    let mut m: M3 = [[0.; 3]; 3];
    let mut any = false;
    for &t in tris {
        for &u in ctx.neighbours(t) {
            if u > t && inside.contains(&u) {
                let c = ctx.normal(t).cross(ctx.normal(u));
                if c.len() > 1e-4
                    && let Some(d) = c.normalized()
                {
                    outer(&mut m, d, ctx.area(t).min(ctx.area(u)));
                    any = true;
                }
            }
        }
    }
    if !any {
        return None;
    }
    eigen(m).get(2).map(|e| e.1)
}

/// A circle through points projected along `axis`: (centre, radius).
fn fit_circle(pts: &[Vec3], axis: Vec3) -> Option<(Vec3, f64)> {
    let u = axis.any_perp().normalized()?;
    let v = axis.cross(u);
    let origin = pts.iter().fold(Vec3::ZERO, |s, p| s + *p) / pts.len().max(1) as f64;
    // Kåsa: x² + y² + D x + E y + F = 0, least squares.
    let mut a = [[0.; 3]; 3];
    let mut b = [0.; 3];
    for p in pts {
        let w = *p - origin;
        let (x, y) = (w.dot(u), w.dot(v));
        let row = [x, y, 1.];
        let rhs = -(x * x + y * y);
        for (i, ri) in row.iter().enumerate() {
            for (j, rj) in row.iter().enumerate() {
                if let Some(c) = a.get_mut(i).and_then(|r: &mut [f64; 3]| r.get_mut(j)) {
                    *c += ri * rj;
                }
            }
            if let Some(c) = b.get_mut(i) {
                *c += ri * rhs;
            }
        }
    }
    let [d, e, f] = solve(a, b)?;
    let (cx, cy) = (-d / 2., -e / 2.);
    let r2 = cx * cx + cy * cy - f;
    (r2 > 0.).then(|| (origin + u * cx + v * cy, r2.sqrt()))
}

fn fit_cylinder(ctx: &Ctx, tris: &[u32]) -> Option<Primitive> {
    let axis = ruling_axis(ctx, tris)?;
    let pts = vertices(ctx, tris);
    if pts.len() < 4 {
        return None;
    }
    let (point, radius) = fit_circle(&pts, axis)?;
    Some(Primitive::Cylinder { point, axis, radius })
}

fn fit_sphere(ctx: &Ctx, tris: &[u32]) -> Option<Primitive> {
    let pts = vertices(ctx, tris);
    if pts.len() < 5 {
        return None;
    }
    let origin = pts.iter().fold(Vec3::ZERO, |s, p| s + *p) / pts.len() as f64;
    let mut a = [[0.; 4]; 4];
    let mut b = [0.; 4];
    for p in &pts {
        let w = *p - origin;
        let row = [w.x, w.y, w.z, 1.];
        let rhs = -w.len2();
        for (i, ri) in row.iter().enumerate() {
            for (j, rj) in row.iter().enumerate() {
                if let Some(c) = a.get_mut(i).and_then(|r: &mut [f64; 4]| r.get_mut(j)) {
                    *c += ri * rj;
                }
            }
            if let Some(c) = b.get_mut(i) {
                *c += ri * rhs;
            }
        }
    }
    let [d, e, f, g] = solve(a, b)?;
    let c = Vec3::new(-d / 2., -e / 2., -f / 2.);
    let r2 = c.len2() - g;
    (r2 > 0.).then(|| Primitive::Sphere { center: origin + c, radius: r2.sqrt() })
}

fn fit_cone(ctx: &Ctx, tris: &[u32]) -> Option<Primitive> {
    // The normals of a cone lie on a circle of the unit sphere around its axis.
    let total: f64 = tris.iter().map(|t| ctx.area(*t)).sum();
    if !(total > 0.) {
        return None;
    }
    let mean = tris.iter().fold(Vec3::ZERO, |s, t| s + ctx.normal(*t) * ctx.area(*t)) / total;
    let mut m: M3 = [[0.; 3]; 3];
    for &t in tris {
        outer(&mut m, ctx.normal(t) - mean, ctx.area(t));
    }
    let axis = eigen(m).first()?.1;
    // Every facet's plane passes through the apex: least squares over n·(x - p) = 0.
    let mut a: M3 = [[0.; 3]; 3];
    let mut b = Vec3::ZERO;
    for &t in tris {
        let n = ctx.normal(t);
        let w = ctx.area(t);
        outer(&mut a, n, w);
        b += n * (n.dot(ctx.centroid(t)) * w);
    }
    let [x, y, z] = solve(a, [b.x, b.y, b.z])?;
    let apex = Vec3::new(x, y, z);
    let pts = vertices(ctx, tris);
    let mean_h: f64 = pts.iter().map(|p| (*p - apex).dot(axis)).sum::<f64>() / pts.len().max(1) as f64;
    let axis = if mean_h < 0. { -axis } else { axis };
    let angles: Vec<f64> = pts
        .iter()
        .map(|p| {
            let w = *p - apex;
            let h = w.dot(axis);
            (w - axis * h).len().atan2(h)
        })
        .collect();
    let half_angle = angles.iter().sum::<f64>() / angles.len().max(1) as f64;
    // Nearly a cylinder or a plane: those fit better as what they are.
    if !(0.5f64.to_radians()..=89.5f64.to_radians()).contains(&half_angle) {
        return None;
    }
    Some(Primitive::Cone { apex, axis, half_angle })
}

/// A torus: the spine circle is where the facets' normal lines meet it, `minor` in from the
/// surface. Search `minor` (either side) for the offset points that lie best on one circle,
/// then refine everything on the vertices.
fn fit_torus(ctx: &Ctx, tris: &[u32]) -> Option<Primitive> {
    if tris.len() < 24 {
        return None;
    }
    let samples: Vec<(Vec3, Vec3)> = tris.iter().step_by(tris.len().div_ceil(800).max(1)).map(|t| (ctx.centroid(*t), ctx.normal(*t))).collect();
    let mut bounds = solidcraft_geom::Aabb3::EMPTY;
    for (p, _) in &samples {
        bounds.add(*p);
    }
    let extent = bounds.size().len().max(1e-6);
    // The spine through points offset by `r` along their normals: (circle, residual).
    let spine = |r: f64| -> Option<(Vec3, Vec3, f64, f64)> {
        let q: Vec<Vec3> = samples.iter().map(|(p, n)| *p - *n * r).collect();
        let c = q.iter().fold(Vec3::ZERO, |s, p| s + *p) / q.len() as f64;
        let mut m: M3 = [[0.; 3]; 3];
        for p in &q {
            outer(&mut m, *p - c, 1.);
        }
        let axis = eigen(m).first()?.1;
        let (center, radius) = fit_circle(&q, axis)?;
        let residual: f64 = q
            .iter()
            .map(|p| {
                let w = *p - center;
                let h = w.dot(axis);
                ((w - axis * h).len() - radius).powi(2) + h * h
            })
            .sum::<f64>()
            / q.len() as f64;
        Some((center, axis, radius, residual))
    };
    let mut best: Option<(f64, (Vec3, Vec3, f64, f64))> = None;
    for k in 1..=120 {
        let r = extent * 1e-3 * (2000f64).powf(k as f64 / 120.);
        for r in [r, -r] {
            if let Some(f) = spine(r)
                && best.as_ref().is_none_or(|(_, b)| f.3 < b.3)
            {
                best = Some((r, f));
            }
        }
    }
    let (r, (center, axis, major, _)) = best?;
    let minor = r.abs();
    if !(major > minor * 1.0001) || !(minor > 0.) {
        return None;
    }
    Some(Primitive::Torus { center, axis, major, minor })
}

fn fit(ctx: &Ctx, kind: SurfaceKind, tris: &[u32]) -> Option<Primitive> {
    let first = match kind {
        SurfaceKind::Plane | SurfaceKind::Auto => fit_plane(ctx, tris),
        SurfaceKind::Cylinder => fit_cylinder(ctx, tris),
        SurfaceKind::Cone => fit_cone(ctx, tris),
        SurfaceKind::Sphere => fit_sphere(ctx, tris),
        SurfaceKind::Torus => fit_torus(ctx, tris),
    }?;
    Some(refine(first, &vertices(ctx, tris)))
}

/// The surface's parameters as a vector (and back): what least squares adjusts.
fn params(s: &Primitive) -> [f64; 8] {
    match *s {
        Primitive::Plane { point, normal } => [point.x, point.y, point.z, normal.x, normal.y, normal.z, 0., 0.],
        Primitive::Cylinder { point, axis, radius } => [point.x, point.y, point.z, axis.x, axis.y, axis.z, radius, 0.],
        Primitive::Cone { apex, axis, half_angle } => [apex.x, apex.y, apex.z, axis.x, axis.y, axis.z, half_angle, 0.],
        Primitive::Sphere { center, radius } => [center.x, center.y, center.z, radius, 0., 0., 0., 0.],
        Primitive::Torus { center, axis, major, minor } => [center.x, center.y, center.z, axis.x, axis.y, axis.z, major, minor],
    }
}

fn with_params(s: &Primitive, p: &[f64; 8]) -> Option<Primitive> {
    let [x, y, z, a, b, c, d, e] = *p;
    let point = Vec3::new(x, y, z);
    let dir = || Vec3::new(a, b, c).normalized();
    let out = match s {
        Primitive::Plane { .. } => Primitive::Plane { point, normal: dir()? },
        Primitive::Cylinder { .. } => Primitive::Cylinder { point, axis: dir()?, radius: d },
        Primitive::Cone { .. } => Primitive::Cone { apex: point, axis: dir()?, half_angle: d },
        Primitive::Sphere { .. } => Primitive::Sphere { center: point, radius: a },
        Primitive::Torus { .. } => Primitive::Torus { center: point, axis: dir()?, major: d, minor: e },
    };
    params(&out).iter().all(|v| v.is_finite()).then_some(out)
}

/// Least squares on the vertices' distances (Levenberg-Marquardt): the facet-based first fit
/// of a coarse mesh is off by a fraction of a degree; this brings it onto the vertices.
fn refine(s: Primitive, pts: &[Vec3]) -> Primitive {
    if matches!(s, Primitive::Plane { .. }) || pts.len() < 6 {
        return s;
    }
    // A torus adjusts all eight parameters; the other surfaces the first seven, as they always
    // have (a sphere's unused ones leave the system singular, so it keeps its linear fit).
    if matches!(s, Primitive::Torus { .. }) { refine_n::<8>(s, pts) } else { refine_n::<7>(s, pts) }
}

fn refine_n<const N: usize>(s: Primitive, pts: &[Vec3]) -> Primitive {
    let step = pts.len().div_ceil(1500).max(1);
    let sample: Vec<Vec3> = pts.iter().step_by(step).copied().collect();
    let cost = |s: &Primitive| sample.iter().map(|p| s.signed_distance(*p).powi(2)).sum::<f64>();
    let mut best = s;
    let mut best_cost = cost(&s);
    let mut lambda = 1e-3;
    for _ in 0..30 {
        let all = params(&best);
        let mut p0 = [0.; N];
        for (v, a) in p0.iter_mut().zip(all) {
            *v = a;
        }
        let full = |q: &[f64; N]| {
            let mut out = all;
            for (o, v) in out.iter_mut().zip(q) {
                *o = *v;
            }
            out
        };
        let scale = p0.iter().take(3).map(|v| v.abs()).fold(1f64, f64::max);
        let mut jt_j = [[0.; N]; N];
        let mut jt_r = [0.; N];
        let mut columns: Vec<Vec<f64>> = Vec::with_capacity(N);
        for k in 0..N {
            let h = if k < 3 { 1e-7 * scale } else { 1e-7 };
            let mut q = p0;
            if let Some(v) = q.get_mut(k) {
                *v += h;
            }
            let col: Vec<f64> = match with_params(&best, &full(&q)) {
                Some(t) => sample.iter().map(|p| (t.signed_distance(*p) - best.signed_distance(*p)) / h).collect(),
                None => vec![0.; sample.len()],
            };
            columns.push(col);
        }
        let residual: Vec<f64> = sample.iter().map(|p| best.signed_distance(*p)).collect();
        for i in 0..N {
            for j in 0..N {
                let (Some(ci), Some(cj)) = (columns.get(i), columns.get(j)) else { continue };
                if let Some(c) = jt_j.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *c = ci.iter().zip(cj).map(|(a, b)| a * b).sum();
                }
            }
            if let (Some(ci), Some(c)) = (columns.get(i), jt_r.get_mut(i)) {
                *c = -ci.iter().zip(&residual).map(|(a, b)| a * b).sum::<f64>();
            }
        }
        let mut improved = false;
        for _ in 0..8 {
            let mut a = jt_j;
            for (i, row) in a.iter_mut().enumerate() {
                if let Some(d) = row.get_mut(i) {
                    *d += lambda * (*d).max(1e-12) + 1e-15;
                }
            }
            let Some(delta) = solve(a, jt_r) else { break };
            let mut q = p0;
            for (v, d) in q.iter_mut().zip(delta) {
                *v += d;
            }
            if let Some(t) = with_params(&best, &full(&q)) {
                let c = cost(&t);
                if c < best_cost {
                    best = t;
                    best_cost = c;
                    lambda = (lambda * 0.3).max(1e-12);
                    improved = true;
                    break;
                }
            }
            lambda *= 10.;
        }
        if !improved || best_cost < 1e-24 {
            break;
        }
    }
    best
}

/// Largest distance of the triangles' corners from `s`, or infinity when a facet faces away.
fn deviation(ctx: &Ctx, s: &Primitive, tris: &[u32], tol: f64) -> f64 {
    let mut worst = 0f64;
    for &t in tris {
        if !ctx.on(t, s, f64::INFINITY) {
            return f64::INFINITY;
        }
        for p in ctx.corners(t) {
            worst = worst.max(s.signed_distance(p).abs());
        }
        if worst > tol * 1e3 {
            return worst;
        }
    }
    worst
}

/// The seed and its smooth neighbourhood (folds under [`SMOOTH_FOLD`]), nearest first.
fn neighbourhood(ctx: &Ctx, seed: u32, free: &dyn Fn(u32) -> bool, limit: usize) -> Vec<u32> {
    let mut near = vec![seed];
    let mut seen = std::collections::BTreeSet::from([seed]);
    let mut k = 0;
    while let Some(&t) = near.get(k) {
        if near.len() >= limit {
            break;
        }
        k += 1;
        for &u in ctx.neighbours(t) {
            if free(u) && ctx.normal(t).dot(ctx.normal(u)) >= SMOOTH_FOLD.to_radians().cos() && seen.insert(u) {
                near.push(u);
            }
        }
    }
    near.truncate(limit);
    near
}

/// Grow a region from `seed` over triangles not yet `claimed`.
///
/// The kind is chosen on the seed's smooth neighbourhood: the simplest kind that fits the
/// largest part of it (so a cone is not taken for a narrow cylinder, nor a bore for a strip of
/// planes). The region then grows over every neighbour on that surface, refitting as it grows.
pub(crate) fn grow(ctx: &Ctx, seed: u32, want: SurfaceKind, tol: f64, claimed: &[bool]) -> Option<MeshRegion> {
    let n = ctx.triangles.len();
    let free = |t: u32| !claimed.get(t as usize).copied().unwrap_or(true);
    if !free(seed) {
        return None;
    }
    let kinds: &[SurfaceKind] = match want {
        SurfaceKind::Auto => &[SurfaceKind::Plane, SurfaceKind::Cylinder, SurfaceKind::Cone, SurfaceKind::Sphere, SurfaceKind::Torus],
        SurfaceKind::Plane => &[SurfaceKind::Plane],
        SurfaceKind::Cylinder => &[SurfaceKind::Cylinder],
        SurfaceKind::Cone => &[SurfaceKind::Cone],
        SurfaceKind::Sphere => &[SurfaceKind::Sphere],
        SurfaceKind::Torus => &[SurfaceKind::Torus],
    };
    let near = neighbourhood(ctx, seed, &free, 96);
    let mut start: Option<(Primitive, usize)> = None;
    let mut size = near.len();
    while start.is_none() && size >= 1 {
        let part = near.get(..size)?;
        for kind in kinds {
            if let Some(s) = fit(ctx, *kind, part)
                && deviation(ctx, &s, part, tol) <= tol
            {
                start = Some((s, size));
                break;
            }
        }
        size = if size > 4 { size * 3 / 4 } else { size - 1 };
    }
    let (mut surface, size) = match start {
        Some(s) => s,
        // A lone facet of a fine mesh is a plane of its own.
        None if kinds.contains(&SurfaceKind::Plane) => {
            let [a, ..] = ctx.corners(seed);
            (Primitive::Plane { point: a, normal: ctx.normal(seed) }, 1)
        }
        None => return None,
    };
    let mut in_patch = vec![false; n];
    let mut patch: Vec<u32> = near.get(..size)?.to_vec();
    for &t in &patch {
        if let Some(p) = in_patch.get_mut(t as usize) {
            *p = true;
        }
    }
    let mut fitted_at = patch.len();
    for _ in 0..64 {
        let mut queue: VecDeque<u32> = patch.iter().copied().collect();
        let mut added = 0usize;
        while let Some(t) = queue.pop_front() {
            for &u in ctx.neighbours(t) {
                if !free(u) || in_patch.get(u as usize).copied().unwrap_or(true) || !ctx.on(u, &surface, tol) {
                    continue;
                }
                if let Some(p) = in_patch.get_mut(u as usize) {
                    *p = true;
                }
                patch.push(u);
                queue.push_back(u);
                added += 1;
                // Follow the surface: refit as the region doubles.
                if patch.len() >= fitted_at * 2
                    && let Some(s) = fit(ctx, surface.kind(), &patch)
                    && deviation(ctx, &s, &patch, tol) <= tol
                {
                    surface = s;
                    fitted_at = patch.len();
                }
            }
        }
        let refit = fit(ctx, surface.kind(), &patch).filter(|s| deviation(ctx, s, &patch, tol) <= tol);
        match refit {
            Some(s) if s != surface && added > 0 => {
                surface = s;
                fitted_at = patch.len();
                continue;
            }
            Some(s) => surface = s,
            None => {}
        }
        // Stopped: does a richer kind fit the region with its smooth neighbours (a fillet that
        // looked like a cylinder over its first few facets is a torus all round)?
        // A plane stops where its flat face ends; only curved regions move on.
        if want != SurfaceKind::Auto || surface.kind() == SurfaceKind::Plane {
            break;
        }
        let mut ring: Vec<u32> = Vec::new();
        for &t in &patch {
            for &u in ctx.neighbours(t) {
                if free(u) && !in_patch.get(u as usize).copied().unwrap_or(true) && ctx.normal(t).dot(ctx.normal(u)) >= SMOOTH_FOLD.to_radians().cos()
                {
                    ring.push(u);
                }
            }
        }
        ring.sort_unstable();
        ring.dedup();
        if ring.is_empty() {
            break;
        }
        let rank = |k: SurfaceKind| kinds.iter().position(|x| *x == k).unwrap_or(0);
        let mut grown = patch.clone();
        grown.extend_from_slice(&ring);
        let richer = kinds.iter().skip(rank(surface.kind()) + 1).find_map(|k| fit(ctx, *k, &grown).filter(|s| deviation(ctx, s, &grown, tol) <= tol));
        let Some(s) = richer else { break };
        surface = s;
        for u in &ring {
            if let Some(p) = in_patch.get_mut(*u as usize) {
                *p = true;
            }
        }
        patch = grown;
        fitted_at = patch.len();
    }
    let pts = vertices(ctx, &patch);
    let (mut worst, mut sum) = (0f64, 0f64);
    for p in &pts {
        let d = surface.signed_distance(*p);
        worst = worst.max(d.abs());
        sum += d * d;
    }
    let rms = (sum / pts.len().max(1) as f64).sqrt();
    let concave = match surface {
        Primitive::Plane { .. } => false,
        _ => {
            let facing: f64 = patch.iter().map(|t| ctx.normal(*t).dot(surface.normal_at(ctx.centroid(*t))) * ctx.area(*t)).sum();
            facing < 0.
        }
    };
    let area = patch.iter().map(|t| ctx.area(*t)).sum();
    patch.sort_unstable();
    Some(MeshRegion { surface, triangles: patch, area, max_deviation: worst, rms_deviation: rms, concave })
}

/// Design intent over measurement noise: directions (plane normals, axes) within half a degree
/// of each other become one, those within half a degree of a model axis or of square to a
/// larger one become exactly that, and parallel planes within the tolerance become one plane.
fn regularize(regions: &mut [MeshRegion], tolerance: f64) {
    let close = 0.5f64.to_radians();
    let direction = |s: &Primitive| match *s {
        Primitive::Plane { normal, .. } => Some(normal),
        Primitive::Cylinder { axis, .. } | Primitive::Cone { axis, .. } | Primitive::Torus { axis, .. } => Some(axis),
        Primitive::Sphere { .. } => None,
    };
    // Directions by the area behind them, largest first.
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by(|a, b| regions.get(*b).map_or(0., |r| r.area).total_cmp(&regions.get(*a).map_or(0., |r| r.area)));
    let mut kept: Vec<Vec3> = Vec::new();
    for &i in &order {
        let Some(d) = regions.get(i).and_then(|r| direction(&r.surface)) else { continue };
        let mut e = kept.iter().copied().find(|k| k.dot(d).abs() >= close.cos()).map(|k| if k.dot(d) < 0. { -k } else { k });
        if e.is_none() {
            let mut c = [Vec3::X, Vec3::Y, Vec3::Z]
                .into_iter()
                .find(|a| a.dot(d).abs() >= close.cos())
                .map(|a| if a.dot(d) < 0. { -a } else { a })
                .unwrap_or(d);
            // Square to a larger direction when nearly so.
            for k in &kept {
                if k.dot(c).abs() <= close.sin()
                    && let Some(sq) = (c - *k * k.dot(c)).normalized()
                {
                    c = sq;
                }
            }
            kept.push(c);
            e = Some(c);
        }
        let Some(e) = e else { continue };
        if let Some(r) = regions.get_mut(i) {
            r.surface = match r.surface {
                Primitive::Plane { point, .. } => Primitive::Plane { point, normal: e },
                Primitive::Cylinder { point, radius, .. } => Primitive::Cylinder { point, axis: e, radius },
                Primitive::Cone { apex, half_angle, .. } => Primitive::Cone { apex, axis: e, half_angle },
                Primitive::Torus { center, major, minor, .. } => Primitive::Torus { center, axis: e, major, minor },
                other => other,
            };
        }
    }
    // Coplanar planes: one offset, weighted by area.
    let n = regions.len();
    for i in 0..n {
        let Some(Primitive::Plane { point: pi, normal: ni }) = regions.get(i).map(|r| r.surface) else { continue };
        let mut members = vec![i];
        for j in i + 1..n {
            if let Some(Primitive::Plane { point: pj, normal: nj }) = regions.get(j).map(|r| r.surface)
                && nj.dot(ni).abs() >= 1. - 1e-12
                && (pj - pi).dot(ni).abs() <= tolerance
            {
                members.push(j);
            }
        }
        if members.len() < 2 {
            continue;
        }
        let (mut sum, mut weight) = (0., 0.);
        for m in &members {
            if let Some(r) = regions.get(*m)
                && let Primitive::Plane { point, .. } = r.surface
            {
                sum += (point - pi).dot(ni) * r.area;
                weight += r.area;
            }
        }
        let offset = if weight > 0. { sum / weight } else { 0. };
        for m in &members {
            if let Some(r) = regions.get_mut(*m)
                && let Primitive::Plane { point, normal } = r.surface
            {
                let along = (point - pi).dot(ni);
                r.surface = Primitive::Plane { point: point + ni * (offset - along), normal };
            }
        }
    }
}

/// The picked regions of a welded `mesh`, with every vertex on them put onto its surface
/// (onto where the surfaces meet, for a vertex on several), and the picked cones and spheres
/// as features for [`crate::mesh_convert_milled`]. A vertex the surfaces would move by more
/// than twice the tolerance stays where it is; the certification of the result then decides.
pub(crate) fn snap_to_picks(
    mesh: &TriMesh,
    tolerance: f64,
    picks: &[(Vec3, SurfaceKind)],
) -> Result<(TriMesh, crate::mesh_convert_milled::Features)> {
    let ctx = Ctx::new(mesh)?;
    let n = ctx.triangles.len();
    let mut claimed = vec![false; n];
    let mut regions: Vec<MeshRegion> = Vec::new();
    for (i, (point, kind)) in picks.iter().enumerate() {
        let seed = ctx.nearest(*point).ok_or_else(|| invalid("empty mesh"))?;
        if claimed.get(seed as usize).copied().unwrap_or(true) {
            // A second pick on a region already recognised.
            continue;
        }
        let region = grow(&ctx, seed, *kind, tolerance, &claimed).ok_or_else(|| {
            let what = match kind {
                SurfaceKind::Auto => "no plane, cylinder, cone, sphere or torus",
                SurfaceKind::Plane => "no plane",
                SurfaceKind::Cylinder => "no cylinder",
                SurfaceKind::Cone => "no cone",
                SurfaceKind::Sphere => "no sphere",
                SurfaceKind::Torus => "no torus",
            };
            invalid(&format!("pick {}: {what} fits the mesh there within {tolerance} mm", i + 1))
        })?;
        for t in &region.triangles {
            if let Some(c) = claimed.get_mut(*t as usize) {
                *c = true;
            }
        }
        regions.push(region);
    }
    regularize(&mut regions, tolerance);
    let mut on: Vec<Vec<usize>> = vec![Vec::new(); mesh.positions.len()];
    for (r, region) in regions.iter().enumerate() {
        for t in &region.triangles {
            for v in ctx.triangles.get(*t as usize).copied().unwrap_or([0, 0, 0]) {
                if let Some(list) = on.get_mut(v as usize)
                    && !list.contains(&r)
                {
                    list.push(r);
                }
            }
        }
    }
    let mut positions = mesh.positions.clone();
    for (p, surfaces) in positions.iter_mut().zip(&on) {
        // Cones and spheres are cut as solids of their own; their vertices need not move.
        if surfaces.is_empty()
            || surfaces.iter().any(|r| regions.get(*r).is_none_or(|g| matches!(g.surface, Primitive::Cone { .. } | Primitive::Sphere { .. })))
        {
            continue;
        }
        let start = *p;
        let mut q = start;
        // Alternating projections settle on the surfaces' common points.
        for _ in 0..12 {
            for r in surfaces {
                if let Some(region) = regions.get(*r) {
                    q = region.surface.project(q);
                }
            }
        }
        if q.is_finite() && q.dist(start) <= tolerance * 2. {
            *p = q;
        }
    }
    // A facet the moves would flatten keeps its corners where they were.
    for _ in 0..4 {
        let mut reverted = false;
        for t in &mesh.triangles {
            let [a, b, c] = t.map(|i| positions.get(i as usize).copied().unwrap_or(Vec3::ZERO));
            if (b - a).cross(c - a).len() <= 1e-12 * (1. + a.len2()) {
                for i in t {
                    if let (Some(p), Some(o)) = (positions.get_mut(*i as usize), mesh.positions.get(*i as usize))
                        && *p != *o
                    {
                        *p = *o;
                        reverted = true;
                    }
                }
            }
        }
        if !reverted {
            break;
        }
    }
    // Cones and spheres leave the mesh: each region is replaced by a fan over its rim, which
    // fills a drill point up to its bore's floor and a dimple up to its face. What is left is
    // milled; the cones and spheres are cut from it afterwards.
    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(mesh.triangles.len());
    let mut solids = Vec::new();
    let mut removed = vec![false; n];
    for region in &regions {
        if matches!(region.surface, Primitive::Cone { .. } | Primitive::Sphere { .. }) {
            for t in &region.triangles {
                if let Some(r) = removed.get_mut(*t as usize) {
                    *r = true;
                }
            }
            solids.push((region.surface, vertices(&ctx, &region.triangles), region.concave));
        }
    }
    let mut rim = std::collections::BTreeMap::<(u32, u32), i32>::new();
    for (t, tri) in mesh.triangles.iter().enumerate() {
        if !removed.get(t).copied().unwrap_or(false) {
            triangles.push(*tri);
            continue;
        }
        let [a, b, c] = *tri;
        for (p, q) in [(a, b), (b, c), (c, a)] {
            *rim.entry((p, q)).or_default() += 1;
            *rim.entry((q, p)).or_default() -= 1;
        }
    }
    // Rim edges, in the removed triangles' direction: one fan per loop, around its centre.
    let mut next = std::collections::BTreeMap::<u32, u32>::new();
    for (&(p, q), &k) in &rim {
        if k > 0 && next.insert(p, q).is_some() {
            return Err(invalid("a picked cone or sphere touches itself at a vertex"));
        }
    }
    while let Some((&start, _)) = next.first_key_value() {
        let mut ring = vec![start];
        let mut current = start;
        loop {
            let Some(to) = next.remove(&current) else { return Err(invalid("the rim of a picked cone or sphere is not closed")) };
            if to == start {
                break;
            }
            ring.push(to);
            current = to;
        }
        let centre = ring.iter().fold(Vec3::ZERO, |s, i| s + positions.get(*i as usize).copied().unwrap_or(Vec3::ZERO)) / ring.len().max(1) as f64;
        let c = u32::try_from(positions.len()).map_err(|_| invalid("too many vertices"))?;
        positions.push(centre);
        for (k, a) in ring.iter().enumerate() {
            let b = ring.get((k + 1) % ring.len()).copied().unwrap_or(*a);
            triangles.push([*a, b, c]);
        }
    }
    Ok((TriMesh { positions, triangles }, crate::mesh_convert_milled::Features { solids }))
}

/// One solid per picked surface, for building a part by hand from what the mesh shows (each
/// named after its surface, such as `Cylinder Ø8.000`):
/// - a plane: the material side of it, as a block over the whole mesh (a slab to cut by);
/// - a cylinder or cone: the solid of revolution over the region's length, a little longer at
///   both ends, so a bore's solid cuts through cleanly;
/// - a sphere: the whole ball.
///
/// Each surface is regularised as for [`crate::convert_mesh_with_regions`] (directions within
/// half a degree made exact), so blocks and bores meet square. The mesh is not changed.
pub fn fit_bodies(body: &Body, tolerance: f64, picks: &[(Vec3, SurfaceKind)], surfaces: bool, round: f64) -> Result<Vec<(String, Body)>> {
    crate::guard("fit bodies to mesh surfaces", || {
        checked_tolerance(tolerance)?;
        if picks.is_empty() || picks.len() > 10_000 || picks.iter().any(|(p, _)| !p.is_finite()) {
            return Err(invalid("pick at least one surface, at most 10000, at finite points"));
        }
        let ctx = context(body, tolerance)?;
        let n = ctx.triangles.len();
        let mut regions = picked_regions(&ctx, picks, tolerance)?;
        regularize(&mut regions, tolerance);
        round_sizes(&mut regions, round, tolerance);
        let all = vertices(&ctx, &(0..n).filter_map(|t| u32::try_from(t).ok()).collect::<Vec<_>>());
        let mut bounds = solidcraft_geom::Aabb3::EMPTY;
        for p in &all {
            bounds.add(*p);
        }
        let size = bounds.size().len().max(1.);
        let margin = size * 0.02;
        let mut out = Vec::new();
        if surfaces {
            for region in &regions {
                out.push(surface_sheet(&ctx, region, tolerance)?);
            }
            return Ok(out);
        }
        for region in &regions {
            let own = vertices(&ctx, &region.triangles);
            let (name, solid) = match region.surface {
                Primitive::Plane { point, normal } => {
                    let u = normal.any_perp().normalized().ok_or_else(|| invalid("degenerate plane"))?;
                    let v = normal.cross(u);
                    let plane = solidcraft_geom::Plane::new(point, u, v).ok_or_else(|| invalid("degenerate plane"))?;
                    let (mut lo, mut hi) =
                        (solidcraft_geom::Vec2::new(f64::INFINITY, f64::INFINITY), solidcraft_geom::Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY));
                    let mut depth = 0f64;
                    for p in &all {
                        let q = plane.to_local(*p);
                        lo = solidcraft_geom::Vec2::new(lo.x.min(q.x), lo.y.min(q.y));
                        hi = solidcraft_geom::Vec2::new(hi.x.max(q.x), hi.y.max(q.y));
                        depth = depth.max(-(*p - point).dot(normal));
                    }
                    let (lo, hi) = (lo - solidcraft_geom::Vec2::new(margin, margin), hi + solidcraft_geom::Vec2::new(margin, margin));
                    let rect = solidcraft_geom::Region2 {
                        outer: solidcraft_geom::Loop2::polygon(&[
                            lo,
                            solidcraft_geom::Vec2::new(hi.x, lo.y),
                            hi,
                            solidcraft_geom::Vec2::new(lo.x, hi.y),
                        ]),
                        holes: Vec::new(),
                    };
                    ("Plane".to_string(), crate::extrude_planar_regions(&plane, &[rect], -(depth + margin), 0.)?)
                }
                Primitive::Cylinder { point, axis, radius } => {
                    let (a, b) =
                        own.iter().map(|p| (*p - point).dot(axis)).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), h| (a.min(h), b.max(h)));
                    if !(b > a) {
                        return Err(invalid("a picked cylinder has no length"));
                    }
                    let solid = crate::cylinder(point + axis * (a - margin), axis, radius, b - a + 2. * margin)?;
                    (format!("Cylinder Ø{:.3}", 2. * radius), solid)
                }
                Primitive::Cone { half_angle, .. } => {
                    let solid = crate::mesh_convert_milled::feature_solid(&region.surface, &own)?;
                    (format!("Cone {:.1}°", 2. * half_angle.to_degrees()), solid)
                }
                Primitive::Sphere { center, radius } => (format!("Sphere R{radius:.3}"), crate::sphere(center, radius)?),
                Primitive::Torus { center, axis, major, minor } => {
                    let ring = crate::torus(Vec3::ZERO, major, minor)?;
                    // Turn the model's Z onto the axis, then move to the centre.
                    let turn = Vec3::Z.cross(axis);
                    let angle = turn.len().atan2(Vec3::Z.dot(axis));
                    let ring = crate::transform(&ring, center, Vec3::ZERO, turn.normalized().unwrap_or(Vec3::X), angle)?;
                    (format!("Torus R{major:.3} r{minor:.3}"), ring)
                }
            };
            out.push((name, solid));
        }
        Ok(out)
    })
}

/// The regions of `picks`, each over triangles no earlier pick took (a second pick on a region
/// already found adds nothing).
fn picked_regions(ctx: &Ctx, picks: &[(Vec3, SurfaceKind)], tolerance: f64) -> Result<Vec<MeshRegion>> {
    let mut claimed = vec![false; ctx.triangles.len()];
    let mut regions: Vec<MeshRegion> = Vec::new();
    for (i, (point, kind)) in picks.iter().enumerate() {
        let seed = ctx.nearest(*point).ok_or_else(|| invalid("empty mesh"))?;
        if claimed.get(seed as usize).copied().unwrap_or(true) {
            continue;
        }
        let region = grow(ctx, seed, *kind, tolerance, &claimed)
            .ok_or_else(|| invalid(&format!("pick {}: no surface of that kind fits the mesh there within {tolerance} mm", i + 1)))?;
        for t in &region.triangles {
            if let Some(c) = claimed.get_mut(*t as usize) {
                *c = true;
            }
        }
        regions.push(region);
    }
    Ok(regions)
}

/// The boundary of a planar region in `plane`'s coordinates, fitted as lines, arcs and circles
/// (no facets): outer loops counter-clockwise seen along the normal, holes clockwise.
fn planar_outline(ctx: &Ctx, region: &MeshRegion, plane: &solidcraft_geom::Plane, tolerance: f64) -> Result<Vec<solidcraft_geom::Region2>> {
    use solidcraft_geom::{Loop2, Region2};
    let inside: std::collections::BTreeSet<(u32, u32)> = region
        .triangles
        .iter()
        .flat_map(|t| {
            let [a, b, c] = ctx.triangles.get(*t as usize).copied().unwrap_or([0, 0, 0]);
            [(a, b), (b, c), (c, a)]
        })
        .collect();
    let mut next = std::collections::BTreeMap::<u32, u32>::new();
    for &(a, b) in &inside {
        if !inside.contains(&(b, a)) {
            next.insert(a, b);
        }
    }
    let fit = (tolerance * 0.1).max(1e-9);
    let mut loops: Vec<Loop2> = Vec::new();
    while let Some((&start, _)) = next.first_key_value() {
        let mut points = Vec::new();
        let mut current = start;
        for _ in 0..=ctx.positions.len() {
            let Some(to) = next.remove(&current) else { break };
            points.push(plane.to_local(ctx.positions.get(current as usize).copied().unwrap_or(Vec3::ZERO)));
            current = to;
            if current == start {
                break;
            }
        }
        let points = crate::mesh_convert_milled::drop_collinear(&points, fit * 0.5);
        if points.len() >= 3 {
            loops.push(crate::mesh_convert_milled::fit_profile(&points, tolerance, fit));
        }
    }
    let mut regions: Vec<Region2> = loops.iter().filter(|l| l.signed_area() > 0.).map(|l| Region2 { outer: l.clone(), holes: Vec::new() }).collect();
    for hole in loops.iter().filter(|l| l.signed_area() < 0.) {
        let Some(probe) = hole.segs.first().map(solidcraft_geom::Seg2::start) else { continue };
        let owner = regions.iter_mut().filter(|r| r.outer.contains(probe)).min_by(|a, b| a.outer.signed_area().total_cmp(&b.outer.signed_area()));
        if let Some(r) = owner {
            r.holes.push(hole.clone());
        }
    }
    if regions.is_empty() {
        return Err(invalid("the picked plane has no closed outline"));
    }
    Ok(regions)
}

/// A cylinder or cone region's axial range from its point or apex and its angular range
/// around the axis: (along from, along to, start angle, sweep), the sweep a whole turn when
/// the region goes all round. Angles are in the frame `u`, `v` across the axis.
fn around(points: &[Vec3], origin: Vec3, axis: Vec3, u: Vec3, v: Vec3) -> (f64, f64, f64, f64) {
    let (mut a, mut b) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut angles: Vec<f64> = Vec::with_capacity(points.len());
    for p in points {
        let w = *p - origin;
        let h = w.dot(axis);
        a = a.min(h);
        b = b.max(h);
        angles.push(w.dot(v).atan2(w.dot(u)));
    }
    angles.sort_by(f64::total_cmp);
    let tau = std::f64::consts::TAU;
    let mut gap = (0f64, 0f64);
    for (i, x) in angles.iter().enumerate() {
        let y = angles.get(i + 1).copied().unwrap_or(angles.first().copied().unwrap_or(0.) + tau);
        if y - x > gap.1 - gap.0 {
            gap = (*x, y);
        }
    }
    if gap.1 - gap.0 < 30f64.to_radians() { (a, b, 0., tau) } else { (a, b, gap.1, tau - (gap.1 - gap.0)) }
}

/// A sheet on a picked surface, bounded as its region is: a plane's face inside its fitted
/// outline, a cylinder's wall over its length and the angle it covers. Cones and spheres have
/// no sheet yet.
fn surface_sheet(ctx: &Ctx, region: &MeshRegion, tolerance: f64) -> Result<(String, Body)> {
    use solidcraft_geom::{Seg2, Vec2};
    let own = vertices(ctx, &region.triangles);
    match region.surface {
        Primitive::Plane { point, normal } => {
            let plane = solidcraft_geom::Plane::from_normal(point, normal).ok_or_else(|| invalid("degenerate plane"))?;
            let outline = planar_outline(ctx, region, &plane, tolerance)?;
            let mut sheets = outline.iter().map(|r| crate::patch_region(&plane, r)).collect::<Result<Vec<_>>>()?;
            let first = sheets.pop().ok_or_else(|| invalid("the picked plane has no closed outline"))?;
            let sheet = sheets.into_iter().try_fold(first, |acc, s| crate::union_apart(&[acc.clone(), s]).map(|u| u.unwrap_or(acc)))?;
            Ok(("Plane surface".to_string(), sheet))
        }
        Primitive::Cylinder { point, axis, radius } => {
            let u = axis.any_perp().normalized().ok_or_else(|| invalid("degenerate axis"))?;
            let v = axis.cross(u);
            let (a, b, start, sweep) = around(&own, point, axis, u, v);
            if !(b > a) {
                return Err(invalid("a picked cylinder has no length"));
            }
            let plane = solidcraft_geom::Plane::new(point + axis * a, u, v).ok_or_else(|| invalid("degenerate axis"))?;
            let chain: Vec<Seg2> = if sweep >= std::f64::consts::TAU {
                solidcraft_geom::Loop2::circle(Vec2::new(0., 0.), radius).segs
            } else {
                vec![Seg2::Arc { center: Vec2::new(0., 0.), radius, start, sweep }]
            };
            Ok((format!("Cylinder surface Ø{:.3}", 2. * radius), crate::extrude_sheet(&plane, &[chain], 0., b - a)?))
        }
        Primitive::Cone { .. } => Err(invalid("cone surfaces are not supported yet; make the cone a body instead")),
        Primitive::Sphere { .. } => Err(invalid("sphere surfaces are not supported yet; make the sphere a body instead")),
        Primitive::Torus { .. } => Err(invalid("torus surfaces are not supported yet; make the torus a body instead")),
    }
}

/// A sketch's worth of the picked surface, without the mesh's facets: a plane's outline (lines,
/// arcs and circles) in that plane, or a cylinder's or cone's circle in the plane across its
/// axis at the end of its region nearer the axis origin. The plane's frame comes with it; its
/// `x`, `y` are the sketch's axes.
pub fn surface_outline(
    body: &Body,
    point: Vec3,
    kind: SurfaceKind,
    tolerance: f64,
    round: f64,
) -> Result<(solidcraft_geom::Plane, Vec<solidcraft_geom::Region2>)> {
    use solidcraft_geom::{Loop2, Region2, Vec2};
    crate::guard("outline a mesh surface", || {
        checked_tolerance(tolerance)?;
        if !point.is_finite() {
            return Err(invalid("the point must be finite"));
        }
        let ctx = context(body, tolerance)?;
        let mut regions = picked_regions(&ctx, &[(point, kind)], tolerance)?;
        regularize(&mut regions, tolerance);
        round_sizes(&mut regions, round, tolerance);
        let region = regions.first().ok_or_else(|| invalid("nothing recognised there"))?;
        let own = vertices(&ctx, &region.triangles);
        match region.surface {
            Primitive::Plane { point, normal } => {
                let plane = solidcraft_geom::Plane::from_normal(point, normal).ok_or_else(|| invalid("degenerate plane"))?;
                let outline = planar_outline(&ctx, region, &plane, tolerance)?;
                Ok((plane, outline))
            }
            Primitive::Cylinder { point, axis, radius } => {
                let (a, ..) = around(&own, point, axis, Vec3::X, Vec3::Y);
                let plane = solidcraft_geom::Plane::from_normal(point + axis * a, axis).ok_or_else(|| invalid("degenerate axis"))?;
                Ok((plane, vec![Region2 { outer: Loop2::circle(Vec2::new(0., 0.), radius), holes: Vec::new() }]))
            }
            Primitive::Cone { apex, axis, half_angle } => {
                let (a, b, ..) = around(&own, apex, axis, Vec3::X, Vec3::Y);
                let at = if a > 1e-9 { a } else { b };
                let plane = solidcraft_geom::Plane::from_normal(apex + axis * at, axis).ok_or_else(|| invalid("degenerate axis"))?;
                Ok((plane, vec![Region2 { outer: Loop2::circle(Vec2::new(0., 0.), at * half_angle.tan()), holes: Vec::new() }]))
            }
            Primitive::Sphere { .. } => Err(invalid("a sphere has no outline to sketch")),
            Primitive::Torus { center, axis, major, .. } => {
                let plane = solidcraft_geom::Plane::from_normal(center, axis).ok_or_else(|| invalid("degenerate axis"))?;
                Ok((plane, vec![Region2 { outer: Loop2::circle(Vec2::new(0., 0.), major), holes: Vec::new() }]))
            }
        }
    })
}

/// The rigid move that puts a mesh square to the model axes: its largest flat face down on XY
/// (its normal along -Z), the largest direction square to that along X, and the corner of its
/// box at the origin (or, with `centred`, the middle of the box at the origin and its bottom on
/// XY). Without a flat face the largest axis (of a bore, a boss) goes along Z. Returned as
/// [`crate::transform`] takes it: (translation, rotation axis, angle in radians), rotating
/// about the origin first.
pub fn align_to_axes(body: &Body, tolerance: f64, centred: bool) -> Result<(Vec3, Vec3, f64)> {
    crate::guard("align a mesh to the model axes", || {
        checked_tolerance(tolerance)?;
        let mut regions = recognize_all(body, tolerance)?;
        regularize_directions_only(&mut regions);
        // Directions by the area behind them; a plane's (with its outward normal) and an axis's.
        let mut votes: Vec<(Vec3, f64, bool)> = Vec::new();
        for r in &regions {
            let (d, flat) = match r.surface {
                Primitive::Plane { normal, .. } => (normal, true),
                Primitive::Cylinder { axis, .. } | Primitive::Cone { axis, .. } | Primitive::Torus { axis, .. } => (axis, false),
                Primitive::Sphere { .. } => continue,
            };
            match votes.iter_mut().find(|v| v.0.dot(d).abs() > 0.5f64.to_radians().cos()) {
                Some(v) => v.1 += r.area,
                None => votes.push((d, r.area, flat)),
            }
        }
        votes.sort_by(|a, b| b.1.total_cmp(&a.1));
        // The largest flat face lies down: its outward normal becomes -Z.
        let largest_plane = regions
            .iter()
            .filter_map(|r| match r.surface {
                Primitive::Plane { normal, .. } => Some((normal, r.area)),
                _ => None,
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let z = match largest_plane {
            Some((n, _)) => -n,
            None => votes.first().map(|v| v.0).ok_or_else(|| invalid("no surface recognised to align by"))?,
        };
        let x = votes
            .iter()
            .map(|v| v.0)
            .find(|d| d.dot(z).abs() < 1f64.to_radians().sin())
            .and_then(|d| (d - z * d.dot(z)).normalized())
            .or_else(|| z.any_perp().normalized())
            .ok_or_else(|| invalid("degenerate direction"))?;
        let y = z.cross(x);
        // Rows x, y, z: the rotation taking the part's frame onto the model's.
        let m = [[x.x, x.y, x.z], [y.x, y.y, y.z], [z.x, z.y, z.z]];
        let get = |i: usize, j: usize| m.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.);
        let trace = get(0, 0) + get(1, 1) + get(2, 2);
        let angle = ((trace - 1.) / 2.).clamp(-1., 1.).acos();
        let axis = if angle < 1e-9 {
            Vec3::Z
        } else if (std::f64::consts::PI - angle).abs() < 1e-6 {
            // A half turn: the axis from the diagonal.
            let d = |i: usize| ((get(i, i) + 1.) / 2.).max(0.).sqrt();
            let mut a = Vec3::new(d(0), d(1), d(2));
            if get(0, 1) < 0. {
                a.y = -a.y;
            }
            if get(0, 2) < 0. {
                a.z = -a.z;
            }
            a.normalized().unwrap_or(Vec3::Z)
        } else {
            Vec3::new(get(2, 1) - get(1, 2), get(0, 2) - get(2, 0), get(1, 0) - get(0, 1)).normalized().unwrap_or(Vec3::Z)
        };
        let mesh = body.mesh.as_ref().ok_or_else(|| invalid("select a mesh body"))?;
        let mut b = solidcraft_geom::Aabb3::EMPTY;
        for p in &mesh.positions {
            b.add(Vec3::new(p.dot(x), p.dot(y), p.dot(z)));
        }
        let t = if centred { Vec3::new(-(b.min.x + b.max.x) / 2., -(b.min.y + b.max.y) / 2., -b.min.z) } else { -b.min };
        Ok((t, axis, angle))
    })
}

/// [`regularize`]'s direction step alone (no plane offsets moved).
fn regularize_directions_only(regions: &mut [MeshRegion]) {
    regularize(regions, 0.);
}

/// Sizes rounded to `grid` mm where that moves them by no more than `tolerance`: radii and
/// minor radii, the offsets of planes square to a model axis, and the positions of axes along
/// a model axis. Design intent: 7.98 measured is 8.00 drawn.
fn round_sizes(regions: &mut [MeshRegion], grid: f64, tolerance: f64) {
    if !(grid > 0. && grid.is_finite()) {
        return;
    }
    let round = |v: f64| {
        let r = (v / grid).round() * grid;
        if (r - v).abs() <= tolerance { r } else { v }
    };
    let on_axis = |d: Vec3| [Vec3::X, Vec3::Y, Vec3::Z].into_iter().any(|a| d.dot(a).abs() > 1. - 1e-12);
    // Points across an axis-aligned direction: their coordinates across it rounded.
    let across = |p: Vec3, d: Vec3| {
        let keep = |c: f64, k: f64| if k.abs() > 1. - 1e-12 { c } else { round(c) };
        if on_axis(d) { Vec3::new(keep(p.x, d.x), keep(p.y, d.y), keep(p.z, d.z)) } else { p }
    };
    for r in regions.iter_mut() {
        r.surface = match r.surface {
            Primitive::Plane { point, normal } if on_axis(normal) => {
                let along = point.dot(normal);
                Primitive::Plane { point: point + normal * (round(along) - along), normal }
            }
            Primitive::Cylinder { point, axis, radius } => Primitive::Cylinder { point: across(point, axis), axis, radius: round(radius) },
            Primitive::Cone { apex, axis, half_angle } => Primitive::Cone { apex: across(apex, axis), axis, half_angle },
            Primitive::Sphere { center, radius } => Primitive::Sphere { center, radius: round(radius) },
            Primitive::Torus { center, axis, major, minor } => {
                Primitive::Torus { center: across(center, axis), axis, major: round(major), minor: round(minor) }
            }
            other => other,
        };
    }
}

/// How far a mesh lies from a solid made from it: every vertex's distance to the solid's
/// surface, negative where the vertex is inside the solid (the solid has material the mesh
/// doesn't), positive where it is outside (the solid lacks material).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Deviation {
    pub max: f64,
    pub mean: f64,
    pub rms: f64,
    /// Share of the vertices within the tolerance, 0..=1.
    pub within: f64,
    pub vertices: usize,
    /// The farthest vertices: (point, signed distance), farthest first.
    pub worst: Vec<(Vec3, f64)>,
}

/// [`Deviation`] of `mesh`'s vertices from `solid`, against `tolerance` mm.
pub fn mesh_deviation(solid: &Body, mesh: &Body, tolerance: f64) -> Result<Deviation> {
    crate::guard("mesh deviation", || {
        checked_tolerance(tolerance)?;
        if solid.is_mesh() {
            return Err(invalid("compare a solid body with the mesh"));
        }
        let input = mesh.mesh.as_ref().ok_or_else(|| invalid("the second body must be a mesh"))?;
        let surface = solid.tessellate_with((tolerance * 0.1).max(1e-4), true)?;
        let inside = surface.inside_index();
        let tris: Vec<[Vec3; 3]> = surface
            .triangles
            .iter()
            .filter_map(|t| {
                Some([*surface.positions.get(t[0] as usize)?, *surface.positions.get(t[1] as usize)?, *surface.positions.get(t[2] as usize)?])
            })
            .collect();
        if tris.is_empty() {
            return Err(invalid("the solid has no surface"));
        }
        // A grid over the solid's triangles for nearest-point queries.
        let mut b = solidcraft_geom::Aabb3::EMPTY;
        for t in &tris {
            for p in t {
                b.add(*p);
            }
        }
        let cell = (b.size().len() / 48.).max(1e-6);
        let key =
            |p: Vec3| (((p.x - b.min.x) / cell).floor() as i64, ((p.y - b.min.y) / cell).floor() as i64, ((p.z - b.min.z) / cell).floor() as i64);
        let mut grid = std::collections::HashMap::<(i64, i64, i64), Vec<u32>>::new();
        for (i, t) in tris.iter().enumerate() {
            let (lo, hi) = t.iter().fold(
                (Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY), Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY)),
                |(lo, hi), p| (lo.min(*p), hi.max(*p)),
            );
            let (a, z) = (key(lo), key(hi));
            if (z.0 - a.0 + 1) * (z.1 - a.1 + 1) * (z.2 - a.2 + 1) > 4096 {
                continue;
            }
            for x in a.0..=z.0 {
                for y in a.1..=z.1 {
                    for w in a.2..=z.2 {
                        grid.entry((x, y, w)).or_default().push(u32::try_from(i).unwrap_or(u32::MAX));
                    }
                }
            }
        }
        let nearest = |p: Vec3| -> f64 {
            let k = key(p);
            let mut best = f64::INFINITY;
            for ring in 0..64i64 {
                if best <= (ring as f64 - 1.).max(0.) * cell {
                    break;
                }
                for x in k.0 - ring..=k.0 + ring {
                    for y in k.1 - ring..=k.1 + ring {
                        for w in k.2 - ring..=k.2 + ring {
                            if (x - k.0).abs().max((y - k.1).abs()).max((w - k.2).abs()) != ring {
                                continue;
                            }
                            for t in grid.get(&(x, y, w)).into_iter().flatten() {
                                if let Some([a, bb, c]) = tris.get(*t as usize) {
                                    best = best.min(closest_on_triangle(p, *a, *bb, *c).dist(p));
                                }
                            }
                        }
                    }
                }
            }
            best
        };
        let step = input.positions.len().div_ceil(20_000).max(1);
        let mut all: Vec<(Vec3, f64)> = Vec::new();
        for p in input.positions.iter().step_by(step) {
            let d = nearest(*p);
            if d.is_finite() {
                all.push((*p, if inside.contains(*p) && d > tolerance * 0.01 { -d } else { d }));
            }
        }
        let n = all.len().max(1) as f64;
        let max = all.iter().map(|(_, d)| d.abs()).fold(0., f64::max);
        let mean = all.iter().map(|(_, d)| d.abs()).sum::<f64>() / n;
        let rms = (all.iter().map(|(_, d)| d * d).sum::<f64>() / n).sqrt();
        let within = all.iter().filter(|(_, d)| d.abs() <= tolerance).count() as f64 / n;
        all.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        all.truncate(10);
        Ok(Deviation { max, mean, rms, within, vertices: n as usize, worst: all })
    })
}

/// The section of a mesh body by `plane` as fitted lines, arcs and circles in its coordinates
/// (no facets), for a sketch on that plane.
pub fn section_outline(body: &Body, plane: &solidcraft_geom::Plane, tolerance: f64) -> Result<Vec<solidcraft_geom::Region2>> {
    crate::guard("section a mesh", || {
        checked_tolerance(tolerance)?;
        let input = body.mesh.as_ref().ok_or_else(|| invalid("select a mesh body"))?;
        let welded = crate::mesh_convert::weld(input, tolerance)?;
        let mesh = welded.mesh.as_ref().ok_or_else(|| invalid("empty mesh"))?;
        let regions = crate::mesh_convert_milled::section_outline(mesh, plane, tolerance)?;
        if regions.is_empty() {
            return Err(invalid("the plane does not cut the mesh"));
        }
        Ok(regions)
    })
}

/// A point on each region [`recognize_all`] finds, kind `Auto`: picks for
/// [`crate::convert_mesh_with_regions`] covering the whole mesh.
pub fn picks_for_all(body: &Body, tolerance: f64) -> Result<Vec<(Vec3, SurfaceKind)>> {
    let regions = recognize_all(body, tolerance)?;
    let input = body.mesh.as_ref().ok_or_else(|| invalid("select a mesh body"))?;
    let centroid = |t: u32| {
        let [a, b, c] = input.triangles.get(t as usize).copied().unwrap_or([0, 0, 0]);
        let p = |i: u32| input.positions.get(i as usize).copied().unwrap_or(Vec3::ZERO);
        (p(a) + p(b) + p(c)) / 3.
    };
    let area = |t: u32| {
        let [a, b, c] = input.triangles.get(t as usize).copied().unwrap_or([0, 0, 0]);
        let p = |i: u32| input.positions.get(i as usize).copied().unwrap_or(Vec3::ZERO);
        (p(b) - p(a)).cross(p(c) - p(a)).len()
    };
    // A facet or two that fits nothing larger is noise, not a surface: its corners are put
    // onto the surfaces around it instead.
    let total: f64 = regions.iter().map(|r| r.area).sum();
    Ok(regions
        .iter()
        .filter(|r| r.triangles.len() > 3 || r.area > total * 0.005)
        .filter_map(|r| r.triangles.iter().copied().max_by(|a, b| area(*a).total_cmp(&area(*b))).map(|t| (centroid(t), SurfaceKind::Auto)))
        .collect())
}

fn checked_tolerance(tolerance: f64) -> Result<()> {
    if !tolerance.is_finite() || !(1e-6..=1.).contains(&tolerance) {
        return Err(invalid("tolerance must be between 0.000001 and 1 mm"));
    }
    Ok(())
}

fn context(body: &Body, tolerance: f64) -> Result<Ctx> {
    let input = body.mesh.as_ref().ok_or_else(|| invalid("select a mesh body"))?;
    let welded = crate::mesh_convert::weld(input, tolerance)?;
    let mesh = welded.mesh.as_ref().ok_or_else(|| invalid("empty mesh"))?;
    Ctx::new(mesh)
}

/// The region around `point` that one surface of `kind` fits within `tolerance` mm.
pub fn recognize_region(body: &Body, point: Vec3, kind: SurfaceKind, tolerance: f64) -> Result<MeshRegion> {
    crate::guard("recognize surface", || {
        checked_tolerance(tolerance)?;
        if !point.is_finite() {
            return Err(invalid("the point must be finite"));
        }
        let ctx = context(body, tolerance)?;
        let seed = ctx.nearest(point).ok_or_else(|| invalid("empty mesh"))?;
        let claimed = vec![false; ctx.triangles.len()];
        grow(&ctx, seed, kind, tolerance, &claimed).ok_or_else(|| {
            let what = match kind {
                SurfaceKind::Auto => "no plane, cylinder, cone, sphere or torus",
                SurfaceKind::Plane => "no plane",
                SurfaceKind::Cylinder => "no cylinder",
                SurfaceKind::Cone => "no cone",
                SurfaceKind::Sphere => "no sphere",
                SurfaceKind::Torus => "no torus",
            };
            invalid(&format!("{what} fits the mesh around that point within {tolerance} mm"))
        })
    })
}

/// The whole mesh as regions, largest triangles first; triangles no surface fits stay out.
pub fn recognize_all(body: &Body, tolerance: f64) -> Result<Vec<MeshRegion>> {
    crate::guard("recognize surfaces", || {
        checked_tolerance(tolerance)?;
        let ctx = context(body, tolerance)?;
        let n = ctx.triangles.len();
        let mut order: Vec<u32> = (0..n).filter_map(|t| u32::try_from(t).ok()).collect();
        order.sort_by(|a, b| ctx.area(*b).total_cmp(&ctx.area(*a)));
        let mut claimed = vec![false; n];
        let mut out = Vec::new();
        for seed in order {
            if claimed.get(seed as usize).copied().unwrap_or(true) || ctx.area(seed) <= 0. {
                continue;
            }
            match grow(&ctx, seed, SurfaceKind::Auto, tolerance, &claimed) {
                Some(region) => {
                    for t in &region.triangles {
                        if let Some(c) = claimed.get_mut(*t as usize) {
                            *c = true;
                        }
                    }
                    out.push(region);
                    if out.len() >= MAX_REGIONS {
                        return Err(invalid(&format!("more than {MAX_REGIONS} regions; pick the surfaces one by one")));
                    }
                }
                None => {
                    if let Some(c) = claimed.get_mut(seed as usize) {
                        *c = true;
                    }
                }
            }
        }
        Ok(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoolOp;

    fn mesh_of(body: &Body) -> Body {
        let mesh = body.tessellate(0.005).unwrap();
        crate::mesh_body(&mesh.positions, &mesh.triangles).unwrap()
    }

    fn cut(a: &Body, b: &Body) -> Body {
        crate::boolean(a, b, BoolOp::Cut).unwrap().unwrap()
    }

    /// A 40 x 30 x 20 block with an 8 mm bore from the top whose bottom is a 118° drill point,
    /// and a spherical dimple of radius 5 in the front face.
    fn part() -> Body {
        let block = crate::box_solid(Vec3::ZERO, Vec3::new(40., 30., 20.)).unwrap();
        let tip_height = 4. / 59f64.to_radians().tan();
        let profile = solidcraft_geom::Region2 {
            outer: solidcraft_geom::Loop2::polygon(&[
                solidcraft_geom::Vec2::new(0., 8. - tip_height),
                solidcraft_geom::Vec2::new(4., 8.),
                solidcraft_geom::Vec2::new(4., 21.),
                solidcraft_geom::Vec2::new(0., 21.),
            ]),
            holes: Vec::new(),
        };
        // The bore with its drill point, revolved in the plane through its axis.
        let plane = solidcraft_geom::Plane::new(Vec3::new(12., 15., 0.), Vec3::X, Vec3::Z).unwrap();
        let bore = crate::revolve(&plane, &[profile], solidcraft_geom::Vec2::new(0., 0.), solidcraft_geom::Vec2::new(0., 1.), std::f64::consts::TAU)
            .unwrap()
            .remove(0);
        let dimple = crate::sphere(Vec3::new(28., -2., 10.), 5.).unwrap();
        let mut p = block;
        for tool in [&bore, &dimple] {
            p = cut(&p, tool);
        }
        p
    }

    #[test]
    fn picks_find_planes_bores_drill_points_and_spheres() {
        let mesh = mesh_of(&part());
        // The top face.
        let top = recognize_region(&mesh, Vec3::new(30., 25., 20.), SurfaceKind::Auto, 0.02).unwrap();
        let Primitive::Plane { point, normal } = top.surface else { panic!("{:?}", top.surface) };
        assert!((normal.z - 1.).abs() < 1e-6 && (point.z - 20.).abs() < 1e-6, "{normal:?} {point:?}");
        // The bore wall: picked on one facet, the whole bore.
        let bore = recognize_region(&mesh, Vec3::new(16., 15., 14.), SurfaceKind::Auto, 0.02).unwrap();
        let Primitive::Cylinder { radius, axis, point } = bore.surface else { panic!("{:?}", bore.surface) };
        assert!((radius - 4.).abs() < 1e-3 && axis.z.abs() > 0.9999, "{radius} {axis:?}");
        assert!((point.x - 12.).abs() < 1e-3 && (point.y - 15.).abs() < 1e-3);
        assert!(bore.concave);
        assert!((bore.area - std::f64::consts::TAU * 4. * 12.).abs() < 1., "{}", bore.area);
        // The drill point.
        let tip = recognize_region(&mesh, Vec3::new(13.5, 15., 7.4), SurfaceKind::Auto, 0.02).unwrap();
        let Primitive::Cone { half_angle, axis, apex } = tip.surface else { panic!("{:?}", tip.surface) };
        assert!((half_angle.to_degrees() - 59.).abs() < 0.1, "{}", half_angle.to_degrees());
        assert!(axis.z > 0.9999 && (apex.z - (8. - 4. / 59f64.to_radians().tan())).abs() < 1e-3, "{axis:?} {apex:?}");
        assert!(tip.concave);
        // The dimple.
        let ball = recognize_region(&mesh, Vec3::new(28., 2.5, 10.5), SurfaceKind::Auto, 0.02).unwrap();
        let Primitive::Sphere { center, radius } = ball.surface else { panic!("{:?}", ball.surface) };
        assert!((radius - 5.).abs() < 1e-3 && center.dist(Vec3::new(28., -2., 10.)) < 1e-3, "{center:?} {radius}");
        assert!(ball.concave);
        // Asking for a kind that does not fit there is refused.
        assert!(recognize_region(&mesh, Vec3::new(30., 25., 20.), SurfaceKind::Sphere, 0.02).is_err());
    }

    #[test]
    fn a_whole_part_segments_into_its_faces() {
        let mesh = mesh_of(&part());
        let regions = recognize_all(&mesh, 0.02).unwrap();
        let count = |k: SurfaceKind| regions.iter().filter(|r| r.surface.kind() == k).count();
        assert_eq!(
            (count(SurfaceKind::Plane), count(SurfaceKind::Cylinder), count(SurfaceKind::Cone), count(SurfaceKind::Sphere)),
            (6, 1, 1, 1),
            "{:?}",
            regions.iter().map(|r| (r.surface.kind(), r.triangles.len())).collect::<Vec<_>>()
        );
        let covered: usize = regions.iter().map(|r| r.triangles.len()).sum();
        assert_eq!(covered, mesh.mesh.as_ref().unwrap().triangles.len());
    }

    #[test]
    fn picked_drill_point_and_dimple_complete_into_a_solid() {
        let part = part();
        let expected = crate::measure(&part).unwrap().volume;
        let mesh = mesh_of(&part);
        // Without the picks the cone and the sphere fit no machining direction.
        assert!(crate::convert_prismatic_mesh(&mesh, 0.02).is_err());
        let picks = [(Vec3::new(13.5, 15., 7.4), SurfaceKind::Auto), (Vec3::new(28., 2.5, 10.5), SurfaceKind::Sphere)];
        let solid = crate::convert_mesh_with_regions(&mesh, 0.02, &picks).unwrap();
        assert!(solid.validity().is_empty());
        let measured = crate::measure(&solid).unwrap();
        assert!((measured.volume - expected).abs() < 0.2, "{} vs {expected}", measured.volume);
        let types = &measured.merged.face_types;
        assert_eq!(types.get("cylinder"), Some(&1), "{types:?}");
        assert_eq!(types.get("sphere"), Some(&1), "{types:?}");
        assert!(types.get("cone").is_some() || types.keys().any(|k| k.contains("cone") || k.contains("revol")), "{types:?}");
        // A pick where nothing of the asked kind fits names it.
        let err = crate::convert_mesh_with_regions(&mesh, 0.02, &[(Vec3::new(30., 25., 20.), SurfaceKind::Cone)]).unwrap_err().to_string();
        assert!(err.contains("pick 1") && err.contains("no cone"), "{err}");
    }

    #[test]
    fn a_noisy_scan_converts_once_its_surfaces_are_recognised() {
        // A milled block whose mesh vertices carry ±0.01 mm of noise, as from a scan: too rough
        // for the milled conversion, exact once every vertex is put onto its recognised surface.
        let block = crate::box_solid(Vec3::ZERO, Vec3::new(40., 30., 20.)).unwrap();
        let pocket = crate::box_solid(Vec3::new(8., 8., 12.), Vec3::new(24., 22., 21.)).unwrap();
        let bore = crate::cylinder(Vec3::new(32., 15., -1.), Vec3::Z, 3., 22.).unwrap();
        let part = cut(&cut(&block, &pocket), &bore);
        let expected = crate::measure(&part).unwrap().volume;
        let m = part.tessellate(0.005).unwrap();
        let mut seed = 12345u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.02
        };
        // Weld first so shared corners move together.
        let welded = crate::mesh_convert::weld(crate::mesh_body(&m.positions, &m.triangles).unwrap().mesh.as_ref().unwrap(), 0.05).unwrap();
        let w = welded.mesh.as_ref().unwrap();
        let positions: Vec<Vec3> = w.positions.iter().map(|p| *p + Vec3::new(noise(), noise(), noise())).collect();
        let noisy = crate::mesh_body(&positions, &w.triangles).unwrap();
        assert!(crate::convert_prismatic_mesh(&noisy, 0.05).is_err());
        let picks = picks_for_all(&noisy, 0.05).unwrap();
        let solid = crate::convert_mesh_with_regions(&noisy, 0.05, &picks).unwrap();
        let measured = crate::measure(&solid).unwrap();
        assert!((measured.volume - expected).abs() < 5., "{} vs {expected}", measured.volume);
        assert_eq!(measured.merged.face_types.get("cylinder"), Some(&1), "{:?}", measured.merged.face_types);
        assert!(measured.merged.face_types.get("plane").copied().unwrap_or(0) <= 12, "{:?}", measured.merged.face_types);
    }

    #[test]
    fn picked_surfaces_become_bodies_to_build_with() {
        let part = part();
        let mesh = mesh_of(&part);
        let picks = [
            (Vec3::new(30., 25., 20.), SurfaceKind::Auto),
            (Vec3::new(16., 15., 14.), SurfaceKind::Auto),
            (Vec3::new(13.5, 15., 7.4), SurfaceKind::Auto),
            (Vec3::new(28., 2.5, 10.5), SurfaceKind::Auto),
        ];
        let bodies = fit_bodies(&mesh, 0.02, &picks, false, 0.).unwrap();
        let names: Vec<&str> = bodies.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["Plane", "Cylinder Ø8.000", "Cone 118.0°", "Sphere R5.000"], "{names:?}");
        // The top face's block lies under z = 20 across the whole part.
        let top = crate::measure(&bodies[0].1).unwrap().bbox;
        assert!((top.max.z - 20.).abs() < 1e-6 && top.min.z < 0. && top.min.x < 0. && top.max.x > 40., "{top:?}");
        // The bore's solid runs past both ends of the bore.
        let bore = crate::measure(&bodies[1].1).unwrap().bbox;
        assert!(bore.max.z > 20. && bore.min.z < 8., "{bore:?}");
        // Cutting the bore and the drill point from the top block leaves the part's hole.
        let cut = crate::boolean(&bodies[0].1, &bodies[1].1, BoolOp::Cut).unwrap().unwrap();
        assert!(crate::measure(&cut).unwrap().volume < crate::measure(&bodies[0].1).unwrap().volume);
        assert!(fit_bodies(&mesh, 0.02, &[], false, 0.).is_err());
        // As surfaces: the top face inside its outline (with the bore's hole), the bore's wall.
        let sheets = fit_bodies(&mesh, 0.02, &picks[..2], true, 0.).unwrap();
        assert_eq!(sheets[0].0, "Plane surface");
        let area = crate::measure(&sheets[0].1).unwrap().area;
        assert!((area - (1200. - std::f64::consts::PI * 16.)).abs() < 0.5, "{area}");
        let wall = crate::measure(&sheets[1].1).unwrap().area;
        assert!((wall - std::f64::consts::TAU * 4. * 12.).abs() < 0.5, "{wall}");
        assert_eq!(crate::measure(&sheets[1].1).unwrap().merged.face_types.get("cylinder"), Some(&1));
        // The top face's outline as a sketch: the rectangle and the bore's circle, no facets.
        let (plane, outline) = surface_outline(&mesh, Vec3::new(30., 25., 20.), SurfaceKind::Auto, 0.02, 0.).unwrap();
        assert!((plane.normal().z - 1.).abs() < 1e-9);
        assert_eq!(outline.len(), 1);
        assert_eq!(outline[0].outer.segs.len(), 4, "{:?}", outline[0].outer.segs);
        assert_eq!(outline[0].holes.len(), 1);
        assert!(
            outline[0].holes[0].segs.iter().all(|s| matches!(s, solidcraft_geom::Seg2::Arc { radius, .. } if (radius - 4.).abs() < 1e-3)),
            "{:?}",
            outline[0].holes[0].segs
        );
        let (_, circle) = surface_outline(&mesh, Vec3::new(16., 15., 14.), SurfaceKind::Auto, 0.02, 0.).unwrap();
        assert!((circle[0].area() - std::f64::consts::PI * 16.).abs() < 1e-3);
    }

    #[test]
    fn fillets_are_recognised_as_tori() {
        // A disc of radius 20 with a 3 mm round on its top edge.
        let disc = crate::cylinder(Vec3::ZERO, Vec3::Z, 20., 10.).unwrap();
        let edges = disc.edges(0.01).unwrap();
        let top: Vec<Vec3> = edges.iter().filter(|e| e.points.iter().all(|p| (p.z - 10.).abs() < 1e-6)).map(|e| e.mid).collect();
        let rounded = crate::fillet(&disc, &top, 3.).unwrap();
        let mesh = mesh_of(&rounded);
        let r = recognize_region(&mesh, Vec3::new(19.1, 0., 9.1), SurfaceKind::Auto, 0.02).unwrap();
        let Primitive::Torus { major, minor, axis, center } = r.surface else { panic!("{:?}", r.surface) };
        assert!((major - 17.).abs() < 0.01 && (minor - 3.).abs() < 0.01, "{major} {minor}");
        assert!(axis.z.abs() > 0.9999 && (center.z - 7.).abs() < 0.01, "{axis:?} {center:?}");
        assert!(!r.concave);
        let bodies = fit_bodies(&mesh, 0.02, &[(Vec3::new(19.1, 0., 9.1), SurfaceKind::Auto)], false, 0.).unwrap();
        assert!(bodies[0].0.starts_with("Torus"), "{}", bodies[0].0);
        let bb = crate::measure(&bodies[0].1).unwrap().bbox;
        assert!((bb.max.z - 10.).abs() < 0.02 && (bb.max.x - 20.).abs() < 0.02, "{bb:?}");
    }

    #[test]
    fn a_turned_mesh_is_put_square_on_the_model_axes() {
        let block = crate::box_solid(Vec3::ZERO, Vec3::new(40., 30., 10.)).unwrap();
        let bore = crate::cylinder(Vec3::new(20., 15., -1.), Vec3::Z, 4., 12.).unwrap();
        let part = cut(&block, &bore);
        let turned = crate::transform(&part, Vec3::new(5., -7., 3.), Vec3::ZERO, Vec3::new(1., 2., 3.), 0.8).unwrap();
        let mesh = mesh_of(&turned);
        let (t, axis, angle) = align_to_axes(&mesh, 0.02, false).unwrap();
        let moved = crate::transform(&mesh, t, Vec3::ZERO, axis, angle).unwrap();
        let bb = crate::measure(&moved).unwrap().bbox;
        let size = bb.size();
        let mut dims = [size.x, size.y, size.z];
        dims.sort_by(f64::total_cmp);
        assert!((bb.min.x).abs() < 1e-6 && bb.min.y.abs() < 1e-6 && bb.min.z.abs() < 1e-6, "{bb:?}");
        assert!((dims[0] - 10.).abs() < 1e-3 && (dims[1] - 30.).abs() < 1e-3 && (dims[2] - 40.).abs() < 1e-3, "{dims:?}");
        assert!((size.z - 10.).abs() < 1e-3, "the largest face lies on XY: {size:?}");
    }

    #[test]
    fn sizes_round_to_the_grid_within_the_tolerance() {
        let block = crate::box_solid(Vec3::ZERO, Vec3::new(40., 30., 10.)).unwrap();
        let bore = crate::cylinder(Vec3::new(20., 15., -1.), Vec3::Z, 3.99, 12.).unwrap();
        let mesh = mesh_of(&cut(&block, &bore));
        let pick = [(Vec3::new(23.99, 15., 5.), SurfaceKind::Auto)];
        let exact = fit_bodies(&mesh, 0.05, &pick, false, 0.).unwrap();
        assert_eq!(exact[0].0, "Cylinder Ø7.980");
        let rounded = fit_bodies(&mesh, 0.05, &pick, false, 0.05).unwrap();
        assert_eq!(rounded[0].0, "Cylinder Ø8.000");
        // Not when rounding would move the size by more than the tolerance.
        let coarse = fit_bodies(&mesh, 0.005, &pick, false, 1.).unwrap();
        assert_eq!(coarse[0].0, "Cylinder Ø7.980");
    }

    #[test]
    fn deviation_measures_a_solid_against_its_mesh() {
        let block = crate::box_solid(Vec3::ZERO, Vec3::new(40., 30., 10.)).unwrap();
        let bore = crate::cylinder(Vec3::new(20., 15., -1.), Vec3::Z, 4., 12.).unwrap();
        let part = cut(&block, &bore);
        let mesh = mesh_of(&part);
        let same = mesh_deviation(&part, &mesh, 0.05).unwrap();
        assert!(same.max < 0.01 && same.within > 0.999, "{same:?}");
        // The block without its bore: the bore's vertices lie inside, 0 .. 4 mm deep.
        let solid = mesh_deviation(&block, &mesh, 0.05).unwrap();
        assert!(solid.max > 3.9 && solid.worst[0].1 < 0., "{:?}", solid.worst.first());
        assert!(solid.within < 0.95);
        assert!(mesh_deviation(&mesh, &mesh, 0.05).is_err());
    }

    #[test]
    fn a_section_sketches_without_facets() {
        let block = crate::box_solid(Vec3::ZERO, Vec3::new(40., 30., 10.)).unwrap();
        let bore = crate::cylinder(Vec3::new(20., 15., -1.), Vec3::Z, 4., 12.).unwrap();
        let mesh = mesh_of(&cut(&block, &bore));
        let plane = solidcraft_geom::Plane::XY.offset(5.);
        let regions = section_outline(&mesh, &plane, 0.02).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].outer.segs.len(), 4);
        assert_eq!(regions[0].holes.len(), 1);
        assert!((regions[0].area() - (1200. - std::f64::consts::PI * 16.)).abs() < 0.01, "{}", regions[0].area());
        assert!(section_outline(&mesh, &solidcraft_geom::Plane::XY.offset(50.), 0.02).is_err());
    }

    #[test]
    fn hostile_inputs_are_refused() {
        let mesh = mesh_of(&crate::box_solid(Vec3::ZERO, Vec3::new(10., 10., 10.)).unwrap());
        for tol in [f64::NAN, 0., -1., 5.] {
            assert!(recognize_region(&mesh, Vec3::ZERO, SurfaceKind::Auto, tol).is_err());
        }
        assert!(recognize_region(&mesh, Vec3::new(f64::NAN, 0., 0.), SurfaceKind::Auto, 0.05).is_err());
        let solid = crate::box_solid(Vec3::ZERO, Vec3::new(10., 10., 10.)).unwrap();
        assert!(recognize_region(&solid, Vec3::ZERO, SurfaceKind::Auto, 0.05).is_err());
    }
}
