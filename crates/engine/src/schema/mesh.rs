//! Contracts and result types for the commands in `cmd/mesh.rs`.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::types::{LengthExpr, Point3};

/// Recognise the analytic surface of a mesh body at a point: the connected region around it
/// that one plane, cylinder, cone or sphere fits within the tolerance, with that surface. With
/// `all`, every surface of the mesh. Changes nothing in the design.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct MeshRecognizeParams {
    /// The mesh body. Default: the selected mesh body.
    #[serde(default)]
    pub body: Option<String>,
    /// A point on the mesh, in mm in the body's coordinates; the triangle nearest it starts the
    /// region. Required unless `all` is true.
    #[serde(default)]
    pub point: Option<Point3>,
    /// The kind of surface: `auto` (whichever fits, simplest first), `plane`, `cylinder`,
    /// `cone` or `sphere`. Default: auto. Ignored with `all`.
    #[serde(default)]
    pub kind: Option<String>,
    /// How far the mesh's vertices may lie from the surface, a length between 0.000001 and
    /// 1 mm. Default: 0.05 mm.
    #[serde(default)]
    pub tolerance: Option<LengthExpr>,
    /// Recognise the whole mesh, largest triangles first, instead of the region at `point`.
    /// Default: false.
    #[serde(default)]
    pub all: Option<bool>,
}

/// One recognised region.
#[derive(Debug, Serialize, JsonSchema)]
pub struct MeshRegionReply {
    /// `plane`, `cylinder`, `cone` or `sphere`.
    pub kind: String,
    /// The surface, in mm and radians: a plane's `point` and `normal` (out of the material), a
    /// cylinder's `point` on its axis, `axis` and `radius`, a cone's `apex`, `axis` (into the
    /// cone) and `half_angle`, a sphere's `center` and `radius`.
    pub surface: Value,
    /// The mesh's triangles (indices) the region covers.
    pub triangles: Vec<u32>,
    /// The region's area, in mm².
    pub area: f64,
    /// Largest and root-mean-square distance of its vertices from the surface, in mm.
    pub max_deviation: f64,
    pub rms_deviation: f64,
    /// The mesh lies on the axis side of a curved surface: a bore or a spherical pocket.
    pub concave: bool,
}

/// What `mesh.recognize` returns.
#[derive(Debug, Serialize, JsonSchema)]
pub struct MeshRecognizeResult {
    /// The mesh body the regions are on.
    pub body: String,
    /// The region at the point, or with `all` every region found, largest first.
    pub regions: Vec<MeshRegionReply>,
    /// In the warn parameter-check mode: what the call's parameters got wrong.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub param_warnings: Vec<String>,
}

#[cfg(test)]
#[path = "mesh_tests.rs"]
mod tests;
