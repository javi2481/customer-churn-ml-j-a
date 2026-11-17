from src.inference.predict import VERSION_PATH


# Esta funcion sirve para comprobar que /health responde ok, con el modelo cargado y su version
def test_health(client):
    # client.get: le pega a /health como si fuera un navegador
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    # la version tiene que ser la que dejo el train en models/, no una fija
    assert data["model_version"] == VERSION_PATH.read_text(encoding="utf-8").strip()
