from ckd_risk.features import FEATURES


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["features"] == FEATURES
    assert "test" in body["metrics"]


def test_predict_separates_sick_and_healthy(client, sick_patient, healthy_patient):
    sick = client.post("/predict", json=sick_patient).json()
    healthy = client.post("/predict", json=healthy_patient).json()
    assert sick["prediction"] == "ckd" and sick["risk_level"] == "high"
    assert healthy["prediction"] == "not_ckd" and healthy["risk_level"] == "low"
    assert sick["missing_features"] == []


def test_predict_handles_missing_values(client):
    r = client.post("/predict", json={"serum_creatinine": 4.5, "haemoglobin": 9.0, "hypertension": "yes"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["ckd_probability"] <= 1
    assert "age" in body["missing_features"] and "serum_creatinine" not in body["missing_features"]


def test_predict_rejects_bad_input(client, sick_patient):
    assert client.post("/predict", json={**sick_patient, "age": -5}).status_code == 422
    assert client.post("/predict", json={**sick_patient, "hypertension": "maybe"}).status_code == 422
    assert client.post("/predict", json={**sick_patient, "unknown_field": 1}).status_code == 422


def test_batch(client, sick_patient, healthy_patient):
    r = client.post("/predict/batch", json={"patients": [sick_patient, healthy_patient]})
    assert [p["prediction"] for p in r.json()["predictions"]] == ["ckd", "not_ckd"]


def test_explain_returns_ranked_contributions(client, sick_patient):
    body = client.post("/explain?top_k=5", json=sick_patient).json()
    contribs = body["contributions"]
    assert len(contribs) == 5
    impacts = [abs(c["shap_value"]) for c in contribs]
    assert impacts == sorted(impacts, reverse=True)
    assert {c["feature"] for c in contribs} <= set(FEATURES)


def test_explain_values_add_up(client, sick_patient):
    # SHAP efficiency: base value + all contributions = predicted probability
    body = client.post(f"/explain?top_k={len(FEATURES)}", json=sick_patient).json()
    total = body["base_value"] + sum(c["shap_value"] for c in body["contributions"])
    assert abs(total - body["ckd_probability"]) < 0.02


def test_explain_flags_imputed_features(client):
    body = client.post(f"/explain?top_k={len(FEATURES)}", json={"serum_creatinine": 4.5}).json()
    flags = {c["feature"]: c["imputed"] for c in body["contributions"]}
    assert flags["serum_creatinine"] is False
    assert flags["age"] is True
