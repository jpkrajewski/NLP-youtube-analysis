import logging
from flask import Flask

from app.routes import views
from app.classifier import ClassiferSingleton

def create_app():
    app = Flask(__name__, template_folder='templates')
    app.register_blueprint(views)
    configure_logging()
    load_classifier()
    return app


def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def load_classifier():
    cs = ClassiferSingleton()
    cs.set_paths(model_path='/app/app/model.sav', vectorizer_path='/app/app/vectorizer.sav')