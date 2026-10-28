//! Parametric reconstruction of imported mechanical mesh bodies.

use serde_json::Value;
use solidcraft_doc::expr::Kind;
use solidcraft_doc::{FeatureKind, MeshPick};
use solidcraft_kernel::SurfaceKind;

use super::{
    CommandSpec,
    features::{add_feature, check_expr},
};
use crate::schema::features::FeatureResult;
use crate::schema::mesh::{MeshFitParams, MeshRecognizeParams, MeshRecognizeResult, MeshRegionReply, MeshSketchParams, MeshSketchResult};
use crate::schema::{TypedContract, output_schema_for};
use crate::{
    Result, Sel, Session,
    params::{bad, bool_, expr, str_, vec3},
};

const RECOGNIZE_CONTRACT: TypedContract<MeshRecognizeParams> = TypedContract::new();
const FIT_CONTRACT: TypedContract<MeshFitParams> = TypedContract::new();
const SKETCH_CONTRACT: TypedContract<MeshSketchParams> = TypedContract::new();

pub static COMMANDS: &[CommandSpec] = &[
    CommandSpec::new("mesh.convert", "Convert Mesh", convert)
        .icon("import")
        .params("body?: mesh body name (default: the selected mesh body); tolerance?: length expression (default 0.05 mm): picks?: [{point: [x,y,z] on the mesh (body coordinates), kind?: auto|plane|cylinder|cone|sphere}] surfaces recognised first (see mesh.recognize), the mesh's vertices put onto them; picked drill points (cones) and spherical pockets are cut as solids of their own; all_surfaces?: bool recognise every surface first (noisy scans); reconstruct a closed constant-section or milled mechanical mesh (walls and holes along up to six machining directions, floors across them) as editable planar, cylindrical, conical and spherical faces, certified against the mesh, with a parametric timeline feature in the mesh body's owning component (the active component must be that component or an ancestor; otherwise activate the named component first); the active component is unchanged; name?"),
    CommandSpec::new("mesh.fit_bodies", "Bodies from Surfaces", crate::checked!("mesh.fit_bodies", fit_bodies))
        .icon("import")
        .contract(&FIT_CONTRACT)
        .output(output_schema_for::<FeatureResult>)
        .params("body?: mesh body name (default: the selected mesh body); tolerance?: length expression (default 0.05 mm); picks?: [{point: [x,y,z] on the mesh (body coordinates), kind?: auto|plane|cylinder|cone|sphere}]; all_surfaces?: bool: one solid per picked surface, to build the part from by hand: a plane's material side as a block over the whole mesh, a cylinder or cone over its region's length (a little longer, so a bore's solid cuts through), a sphere whole; the mesh body stays; name?"),
    CommandSpec::new("mesh.sketch_outline", "Sketch from Surface", crate::checked!("mesh.sketch_outline", sketch_outline))
        .icon("sketch")
        .contract(&SKETCH_CONTRACT)
        .output(output_schema_for::<MeshSketchResult>)
        .params("body?: mesh body name (default: the selected mesh body); point: [x,y,z] on the surface (body coordinates); kind?: auto|plane|cylinder|cone; tolerance?: length expression (default 0.05 mm); name?: a new sketch with the surface's outline and no facets: a plane's boundary as lines, arcs and circles in that plane, a cylinder's or cone's circle in the plane across its axis"),
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

fn fit_bodies(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "mesh.fit_bodies";
    let state = s.model.state();
    let body = target(s, p, cmd)?;
    let source = state.body(&body).ok_or_else(|| bad(cmd, format!("no body `{body}`")))?;
    if !source.body.is_mesh() {
        return Err(bad(cmd, "select a mesh body"));
    }
    let tolerance = expr(p, "tolerance").unwrap_or_else(|| "0.05 mm".into());
    check_expr(s, &tolerance, Kind::Length, cmd, "tolerance")?;
    let picks = picks(p, cmd)?;
    let all_surfaces = bool_(p, "all_surfaces").unwrap_or(false);
    let surfaces = bool_(p, "surfaces").unwrap_or(false);
    if picks.is_empty() && !all_surfaces {
        return Err(bad(cmd, "pick at least one surface on the mesh (or set all_surfaces: true)"));
    }
    let owner = s.doc.model_body_component(source);
    if !s.doc.component_within(owner, s.active_component) {
        let name = s.doc.components.iter().find(|c| c.id == owner).map(|c| c.name.as_str()).unwrap_or(s.doc.name.as_str());
        return Err(bad(cmd, format!("activate component '{name}' first")));
    }
    // The solids belong with the mesh, in its component's own frame.
    let active_component = std::mem::replace(&mut s.active_component, owner);
    let active_occurrence_path = s.active_occurrence_path.take();
    let result = add_feature(s, p, FeatureKind::MeshFit { body: body.into(), tolerance, picks, all_surfaces, surfaces });
    s.active_component = active_component;
    s.active_occurrence_path = active_occurrence_path;
    result
}

fn sketch_outline(s: &mut Session, p: &Value) -> Result<Value> {
    use solidcraft_geom::Seg2;
    use solidcraft_io::Geom2;
    let cmd = "mesh.sketch_outline";
    let state = s.model.state();
    let body = target(s, p, cmd)?;
    let source = state.body(&body).ok_or_else(|| bad(cmd, format!("no body `{body}`")))?;
    if !source.body.is_mesh() {
        return Err(bad(cmd, "select a mesh body"));
    }
    if s.active_sketch.is_some() {
        return Err(bad(cmd, "finish the sketch being edited first"));
    }
    let tolerance = expr(p, "tolerance").unwrap_or_else(|| "0.05 mm".into());
    let tolerance = s.doc.eval(&tolerance, Kind::Length).map_err(|e| bad(cmd, format!("tolerance: {e}")))?;
    let point = p.get("point").and_then(vec3).ok_or_else(|| bad(cmd, "`point` must be [x, y, z] on the mesh"))?;
    let (plane, regions) = solidcraft_kernel::surface_outline(&source.body, point, kind(p, cmd)?, tolerance).map_err(|e| bad(cmd, e.to_string()))?;
    let mut geom = Vec::new();
    for lp in regions.iter().flat_map(|r| std::iter::once(&r.outer).chain(&r.holes)) {
        // A loop of arcs of one circle all round is that circle.
        let circle = match lp.segs.first() {
            Some(Seg2::Arc { center, radius, .. })
                if lp
                    .segs
                    .iter()
                    .all(|x| matches!(x, Seg2::Arc { center: c, radius: r, .. } if c.dist(*center) < 1e-9 && (r - radius).abs() < 1e-9))
                    && (lp
                        .segs
                        .iter()
                        .map(|x| match x {
                            Seg2::Arc { sweep, .. } => sweep.abs(),
                            _ => 0.,
                        })
                        .sum::<f64>()
                        - std::f64::consts::TAU)
                        .abs()
                        < 1e-6 =>
            {
                Some(Geom2::Circle(*center, *radius))
            }
            _ => None,
        };
        if let Some(c) = circle {
            geom.push(c);
            continue;
        }
        for seg in &lp.segs {
            match *seg {
                Seg2::Line { a, b } => geom.push(Geom2::Line(a, b)),
                Seg2::Arc { center, sweep, .. } => {
                    let (a, b) = if sweep > 0. { (seg.start(), seg.end()) } else { (seg.end(), seg.start()) };
                    geom.push(Geom2::Arc { c: center, a, b });
                }
                _ => {}
            }
        }
    }
    let o = plane.origin;
    let mut params = serde_json::json!({
        "text": solidcraft_io::write_dxf(&geom),
        "plane": {"origin": [o.x, o.y, o.z], "x_dir": [plane.x.x, plane.x.y, plane.x.z], "y_dir": [plane.y.x, plane.y.y, plane.y.z]},
        "tolerance": (tolerance * 0.01).clamp(1e-9, 0.01),
    });
    if let Some(n) = str_(p, "name") {
        params["name"] = serde_json::json!(n);
    }
    let made = super::sketch_import::insert(s, &params, cmd, false)?;
    let sketch = made.get("sketch").and_then(Value::as_u64).or(s.active_sketch).unwrap_or(0);
    let curves = made.get("curves").and_then(Value::as_u64).unwrap_or(0) as usize;
    (super::find_command("sketch.finish").ok_or_else(|| bad(cmd, "sketch.finish"))?.run)(s, &serde_json::json!({}))?;
    serde_json::to_value(MeshSketchResult { sketch, curves, param_warnings: Vec::new() }).map_err(|e| bad(cmd, e.to_string()))
}
