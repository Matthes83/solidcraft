use serde_json::json;

use super::{MeshFitParams, MeshRecognizeParams, MeshSketchParams};
use crate::schema::input_schema_for;
use crate::schema::tests::{assert_keys, assert_strict_keeps_todays_errors};

#[test]
fn schemas_and_the_key_check_agree() {
    assert_keys(input_schema_for::<MeshRecognizeParams>(), &["all", "body", "kind", "point", "tolerance"]);
    assert_keys(input_schema_for::<MeshSketchParams>(), &["body", "kind", "name", "point", "tolerance"]);
    assert_keys(
        input_schema_for::<MeshFitParams>(),
        &["all_surfaces", "body", "body_name", "body_names", "name", "participants", "picks", "surfaces", "tolerance"],
    );
}

#[test]
fn strict_keeps_todays_errors() {
    assert_strict_keeps_todays_errors(&[("mesh.recognize", json!({})), ("mesh.recognize", json!({"body": "Missing", "point": [0, 0, 0]}))]);
}
