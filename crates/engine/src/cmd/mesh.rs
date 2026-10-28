//! Parametric reconstruction of imported mechanical mesh bodies.

use serde_json::Value;
use solidcraft_doc::expr::Kind;
use solidcraft_doc::{FeatureKind, MeshPick};
use solidcraft_kernel::SurfaceKind;

use super::{
    CommandSpec,
    features::{add_feature, check_expr},
};
use crate::schema::mesh::{MeshRecognizeParams, MeshRecognizeResult, MeshRegionReply};
use crate::schema::{TypedContract, output_schema_for};
use crate::{
    Result, Sel, Session,
    params::{bad, bool_, expr, str_, vec3},
};

const RECOGNIZE_CONTRACT: TypedContract<MeshRecognizeParams> = TypedContract::new();

pub static COMMANDS: &[CommandSpec] = &[
    CommandSpec::new("mesh.convert", "Convert Mesh", convert)
        .icon("import")
        .params("body?: mesh body name (default: the selected mesh body); tolerance?: length expression (default 0.05 mm): picks?: [{point: [x,y,z] on the mesh (body coordinates), kind?: auto|plane|cylinder|cone|sphere}] surfaces recognised first (see mesh.recognize), the mesh's vertices put onto them; picked drill points (cones) and spherical pockets are cut as solids of their own; all_surfaces?: bool recognise every surface first (noisy scans); reconstruct a closed constant-section or milled mechanical mesh (walls and holes along up to six machining directions, floors across them) as editable planar, cylindrical, conical and spherical faces, certified against the mesh, with a parametric timeline feature in the mesh body's owning component (the active component must be that component or an ancestor; otherwise activate the named component first); the active component is unchanged; name?"),
    CommandSpec::new("mesh.recognize", "Recognize Surface", crate::checked!("mesh.recognize", recognize))
        .icon("inspect")
        .noundo()
        .contract(&RECOGNIZE_CONTRACT)
        .output(output_schema_for::<MeshRecognizeResult>)
        .params("body?: mesh body name (default: the selected mesh body); point?: [x,y,z] on the mesh in the body's coordinates (required unless all); kind?: auto|plane|cylinder|cone|sphere (default auto); tolerance?: length expression (default 0.05 mm); all?: bool recognise every surface: the connected region of the mesh one plane, cylinder, cone or sphere fits within the tolerance, with the surface's parameters and the region's triangles; changes nothing"),
];

/// The mesh body to convert: `body`, else the one selected body.
fn target(s: &Session, p: &Value, cmd: &str) -> Result<String> {
    if let Some(name) = str_(p, "body").filter(|b| !b.trim().is_empty()) {
        return Ok(name.to_owned());
    }
    let selected: Vec<&str> = s.selection.iter().filter_map(|x| if let Sel::Body { name } = x { Some(name.as_str()) } else { None }).collect();
    match selected.as_slice() {
        [name] => Ok((*name).to_owned()),
        [] => Err(bad(cmd, "select a mesh body")),
        _ => Err(bad(cmd, "select one mesh body")),
    }
}

fn convert(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "mesh.convert";
    let state = s.model.state();
    let body = target(s, p, cmd)?;
    let source = state.body(&body).ok_or_else(|| bad(cmd, format!("no body `{body}`")))?;
    if !source.body.is_mesh() {
        return Err(bad(cmd, "select a mesh body"));
    }
    let owner = s.doc.model_body_component(source);
    if !s.doc.component_within(owner, s.active_component) {
        let name = s.doc.components.iter().find(|c| c.id == owner).map(|c| c.name.as_str()).unwrap_or(s.doc.name.as_str());
        return Err(bad(cmd, format!("activate component '{name}' first")));
    }
    let tolerance = expr(p, "tolerance").unwrap_or_else(|| "0.05 mm".into());
    check_expr(s, &tolerance, Kind::Length, cmd, "tolerance")?;
    // Conversion edits the component definition, independent of its occurrence placement.
    let active_component = std::mem::replace(&mut s.active_component, owner);
    let active_occurrence_path = s.active_occurrence_path.take();
    let picks = picks(p, cmd)?;
    let all_surfaces = bool_(p, "all_surfaces").unwrap_or(false);
    let result = add_feature(s, p, FeatureKind::MeshConvert { body: body.into(), tolerance, picks, all_surfaces });
    s.active_component = active_component;
    s.active_occurrence_path = active_occurrence_path;
    result
}

fn kind(p: &Value, cmd: &str) -> Result<SurfaceKind> {
    match p.get("kind") {
        None | Some(Value::Null) => Ok(SurfaceKind::Auto),
        Some(Value::String(k)) => SurfaceKind::parse(k).ok_or_else(|| bad(cmd, "`kind` must be auto, plane, cylinder, cone or sphere")),
        Some(_) => Err(bad(cmd, "`kind` must be auto, plane, cylinder, cone or sphere")),
    }
}

/// `picks`: points on the mesh with the kind of surface at each.
fn picks(p: &Value, cmd: &str) -> Result<Vec<MeshPick>> {
    let Some(v) = p.get("picks").filter(|v| !v.is_null()) else { return Ok(Vec::new()) };
    let list = v.as_array().ok_or_else(|| bad(cmd, "`picks` must be a list of {point, kind}"))?;
    if list.len() > 10_000 {
        return Err(bad(cmd, "at most 10000 picks"));
    }
    list.iter()
        .enumerate()
        .map(|(i, pick)| {
            let point = pick.get("point").and_then(vec3).ok_or_else(|| bad(cmd, format!("pick {}: `point` must be [x, y, z]", i + 1)))?;
            Ok(MeshPick { point: [point.x, point.y, point.z], kind: kind(pick, cmd)? })
        })
        .collect()
}

fn recognize(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "mesh.recognize";
    let state = s.model.state();
    let body = target(s, p, cmd)?;
    let source = state.body(&body).ok_or_else(|| bad(cmd, format!("no body `{body}`")))?;
    if !source.body.is_mesh() {
        return Err(bad(cmd, "select a mesh body"));
    }
    let tolerance = expr(p, "tolerance").unwrap_or_else(|| "0.05 mm".into());
    let tolerance = s.doc.eval(&tolerance, Kind::Length).map_err(|e| bad(cmd, format!("tolerance: {e}")))?;
    let regions = if bool_(p, "all").unwrap_or(false) {
        solidcraft_kernel::recognize_all(&source.body, tolerance)
    } else {
        let point = p.get("point").and_then(vec3).ok_or_else(|| bad(cmd, "`point` must be [x, y, z] on the mesh (or set all: true)"))?;
        solidcraft_kernel::recognize_region(&source.body, point, kind(p, cmd)?, tolerance).map(|r| vec![r])
    }
    .map_err(|e| bad(cmd, e.to_string()))?;
    let regions = regions
        .into_iter()
        .map(|r| MeshRegionReply {
            kind: serde_json::to_value(r.surface.kind()).ok().and_then(|v| v.as_str().map(str::to_owned)).unwrap_or_default(),
            surface: serde_json::to_value(r.surface).unwrap_or(Value::Null),
            triangles: r.triangles,
            area: r.area,
            max_deviation: r.max_deviation,
            rms_deviation: r.rms_deviation,
            concave: r.concave,
        })
        .collect();
    serde_json::to_value(MeshRecognizeResult { body, regions, param_warnings: Vec::new() }).map_err(|e| bad(cmd, e.to_string()))
}
