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
use crate::schema::mesh::{
    MeshAlignParams, MeshAlignResult, MeshDeviationParams, MeshDeviationPoint, MeshDeviationResult, MeshFitParams, MeshRecognizeParams,
    MeshRecognizeResult, MeshRegionReply, MeshSectionParams, MeshSketchParams, MeshSketchResult,
};
use crate::schema::{TypedContract, output_schema_for};
use crate::{
    Result, Sel, Session,
    params::{bad, bool_, expr, str_, vec3},
};

const RECOGNIZE_CONTRACT: TypedContract<MeshRecognizeParams> = TypedContract::new();
const FIT_CONTRACT: TypedContract<MeshFitParams> = TypedContract::new();
const SKETCH_CONTRACT: TypedContract<MeshSketchParams> = TypedContract::new();
const SECTION_CONTRACT: TypedContract<MeshSectionParams> = TypedContract::new();
const ALIGN_CONTRACT: TypedContract<MeshAlignParams> = TypedContract::new();
const DEVIATION_CONTRACT: TypedContract<MeshDeviationParams> = TypedContract::new();

pub static COMMANDS: &[CommandSpec] = &[
    CommandSpec::new("mesh.convert", "Convert Mesh", convert)
        .icon("import")
        .params("body?: mesh body name (default: the selected mesh body); tolerance?: length expression (default 0.05 mm): picks?: [{point: [x,y,z] on the mesh (body coordinates), kind?: auto|plane|cylinder|cone|sphere}] surfaces recognised first (see mesh.recognize), the mesh's vertices put onto them; picked drill points (cones) and spherical pockets are cut as solids of their own; all_surfaces?: bool recognise every surface first (noisy scans); reconstruct a closed constant-section or milled mechanical mesh (walls and holes along up to six machining directions, floors across them) as editable planar, cylindrical, conical and spherical faces, certified against the mesh, with a parametric timeline feature in the mesh body's owning component (the active component must be that component or an ancestor; otherwise activate the named component first); the active component is unchanged; name?"),
    CommandSpec::new("mesh.fit_bodies", "Bodies from Surfaces", crate::checked!("mesh.fit_bodies", fit_bodies))
        .icon("import")
        .contract(&FIT_CONTRACT)
        .output(output_schema_for::<FeatureResult>)
        .params("body?: mesh body name (default: the selected mesh body); tolerance?: length expression (default 0.05 mm); picks?: [{point: [x,y,z] on the mesh (body coordinates), kind?: auto|plane|cylinder|cone|sphere|torus}]; all_surfaces?: bool; surfaces?: bool sheets instead of solids; round?: length grid sizes are rounded to within the tolerance: one solid per picked surface, to build the part from by hand: a plane's material side as a block over the whole mesh, a cylinder or cone over its region's length (a little longer, so a bore's solid cuts through), a sphere whole; the mesh body stays; name?"),
    CommandSpec::new("mesh.sketch_outline", "Sketch from Surface", crate::checked!("mesh.sketch_outline", sketch_outline))
        .icon("sketch")
        .contract(&SKETCH_CONTRACT)
        .output(output_schema_for::<MeshSketchResult>)
        .params("body?: mesh body name (default: the selected mesh body); point: [x,y,z] on the surface (body coordinates); kind?: auto|plane|cylinder|cone|torus; tolerance?: length expression (default 0.05 mm); round?: length grid; name?: a new sketch with the surface's outline and no facets: a plane's boundary as lines, arcs and circles in that plane, a cylinder's or cone's circle in the plane across its axis"),
    CommandSpec::new("mesh.section_sketch", "Section Sketch", crate::checked!("mesh.section_sketch", section_sketch))
        .icon("sketch")
        .contract(&SECTION_CONTRACT)
        .output(output_schema_for::<MeshSketchResult>)
        .params("body?: mesh body name (default: the selected mesh body); plane?: XY|XZ|YZ (default XY); offset?: length expression along the plane's normal (default 0); tolerance?: length expression (default 0.05 mm); name?: a new sketch with the mesh's section by the plane as lines, arcs and circles, without facets"),
    CommandSpec::new("mesh.align", "Align Mesh", crate::checked!("mesh.align", align))
        .icon("move")
        .contract(&ALIGN_CONTRACT)
        .output(output_schema_for::<MeshAlignResult>)
        .params("body?: mesh body name (default: the selected mesh body); centred?: bool (default true: centred on Z, else corner at the origin); tolerance?: length expression (default 0.05 mm): turn and move the mesh square onto the model axes, its largest flat face down on XY (a solid.move)"),
    CommandSpec::new("mesh.deviation", "Mesh Deviation", crate::checked!("mesh.deviation", deviation))
        .icon("inspect")
        .noundo()
        .contract(&DEVIATION_CONTRACT)
        .output(output_schema_for::<MeshDeviationResult>)
        .params("solid: solid body name; body?: mesh body name (default: the selected mesh body); tolerance?: length expression (default 0.05 mm): how far the mesh's vertices lie from the solid (max, mean, rms, share within the tolerance, farthest points; negative inside the solid); changes nothing"),
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
        Some(Value::String(k)) => SurfaceKind::parse(k).ok_or_else(|| bad(cmd, "`kind` must be auto, plane, cylinder, cone, sphere or torus")),
        Some(_) => Err(bad(cmd, "`kind` must be auto, plane, cylinder, cone, sphere or torus")),
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
    let round = expr(p, "round");
    if let Some(r) = &round {
        check_expr(s, r, Kind::Length, cmd, "round")?;
    }
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
    let result = add_feature(s, p, FeatureKind::MeshFit { body: body.into(), tolerance, picks, all_surfaces, surfaces, round });
    s.active_component = active_component;
    s.active_occurrence_path = active_occurrence_path;
    result
}

fn sketch_outline(s: &mut Session, p: &Value) -> Result<Value> {
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
    let round = match expr(p, "round") {
        Some(r) => s.doc.eval(&r, Kind::Length).map_err(|e| bad(cmd, format!("round: {e}")))?,
        None => 0.,
    };
    let (plane, regions) =
        solidcraft_kernel::surface_outline(&source.body, point, kind(p, cmd)?, tolerance, round).map_err(|e| bad(cmd, e.to_string()))?;
    outline_sketch(s, p, cmd, &plane, &regions, tolerance)
}

/// A mesh body named by `body` (or selected), checked to be a mesh.
fn mesh_body(s: &Session, p: &Value, cmd: &str) -> Result<(String, solidcraft_kernel::Body)> {
    let body = target(s, p, cmd)?;
    let state = s.model.state();
    let source = state.body(&body).ok_or_else(|| bad(cmd, format!("no body `{body}`")))?;
    if !source.body.is_mesh() {
        return Err(bad(cmd, "select a mesh body"));
    }
    Ok((body, source.body.clone()))
}

fn tolerance_of(s: &Session, p: &Value, cmd: &str) -> Result<f64> {
    let tolerance = expr(p, "tolerance").unwrap_or_else(|| "0.05 mm".into());
    s.doc.eval(&tolerance, Kind::Length).map_err(|e| bad(cmd, format!("tolerance: {e}")))
}

fn section_sketch(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "mesh.section_sketch";
    let (_, body) = mesh_body(s, p, cmd)?;
    if s.active_sketch.is_some() {
        return Err(bad(cmd, "finish the sketch being edited first"));
    }
    let tolerance = tolerance_of(s, p, cmd)?;
    let name = str_(p, "plane").unwrap_or("XY");
    let mut plane = solidcraft_geom::Plane::named(name).ok_or_else(|| bad(cmd, "`plane` must be XY, XZ or YZ"))?;
    if let Some(o) = expr(p, "offset") {
        let o = s.doc.eval(&o, Kind::Length).map_err(|e| bad(cmd, format!("offset: {e}")))?;
        plane.origin = plane.origin + plane.normal() * o;
    }
    let regions = solidcraft_kernel::section_outline(&body, &plane, tolerance).map_err(|e| bad(cmd, e.to_string()))?;
    outline_sketch(s, p, cmd, &plane, &regions, tolerance)
}

fn align(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "mesh.align";
    let (name, body) = mesh_body(s, p, cmd)?;
    let tolerance = tolerance_of(s, p, cmd)?;
    let centred = bool_(p, "centred").unwrap_or(true);
    let (t, axis, angle) = solidcraft_kernel::align_to_axes(&body, tolerance, centred).map_err(|e| bad(cmd, e.to_string()))?;
    let angle = angle.to_degrees();
    let mut q = serde_json::json!({"bodies": [name], "translate": [t.x, t.y, t.z]});
    if angle.abs() > 1e-9 {
        q["axis"] = serde_json::json!([axis.x, axis.y, axis.z]);
        q["angle"] = serde_json::json!(format!("{angle} deg"));
    }
    (super::find_command("solid.move").ok_or_else(|| bad(cmd, "solid.move"))?.run)(s, &q)?;
    serde_json::to_value(MeshAlignResult {
        body: name,
        axis: [axis.x, axis.y, axis.z],
        angle,
        translate: [t.x, t.y, t.z],
        param_warnings: Vec::new(),
    })
    .map_err(|e| bad(cmd, e.to_string()))
}

fn deviation(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "mesh.deviation";
    let solid = str_(p, "solid").filter(|b| !b.trim().is_empty()).ok_or_else(|| bad(cmd, "`solid` must name a solid body"))?.to_owned();
    let mesh = match str_(p, "body").filter(|b| !b.trim().is_empty()) {
        Some(b) => b.to_owned(),
        None => {
            // The selection may hold the solid too: the mesh is the selected mesh body.
            let state = s.model.state();
            let meshes: Vec<String> = s
                .selection
                .iter()
                .filter_map(|x| if let Sel::Body { name } = x { Some(name.clone()) } else { None })
                .filter(|n| state.body(n).is_some_and(|b| b.body.is_mesh()))
                .collect();
            match meshes.as_slice() {
                [one] => one.clone(),
                [] => return Err(bad(cmd, "select a mesh body")),
                _ => return Err(bad(cmd, "select one mesh body")),
            }
        }
    };
    let tolerance = tolerance_of(s, p, cmd)?;
    let state = s.model.state();
    let a = state.body(&solid).ok_or_else(|| bad(cmd, format!("no body `{solid}`")))?;
    let b = state.body(&mesh).ok_or_else(|| bad(cmd, format!("no body `{mesh}`")))?;
    if !b.body.is_mesh() {
        return Err(bad(cmd, format!("`{mesh}` is not a mesh body")));
    }
    let d = solidcraft_kernel::mesh_deviation(&a.body, &b.body, tolerance).map_err(|e| bad(cmd, e.to_string()))?;
    serde_json::to_value(MeshDeviationResult {
        solid,
        body: mesh,
        max: d.max,
        mean: d.mean,
        rms: d.rms,
        within: d.within,
        vertices: d.vertices,
        worst: d.worst.into_iter().map(|(q, distance)| MeshDeviationPoint { point: [q.x, q.y, q.z], distance }).collect(),
        param_warnings: Vec::new(),
    })
    .map_err(|e| bad(cmd, e.to_string()))
}

/// A new sketch in `plane` with `regions`' loops: circles whole, else lines and arcs.
fn outline_sketch(
    s: &mut Session,
    p: &Value,
    cmd: &str,
    plane: &solidcraft_geom::Plane,
    regions: &[solidcraft_geom::Region2],
    tolerance: f64,
) -> Result<Value> {
    use solidcraft_geom::Seg2;
    use solidcraft_io::Geom2;
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
