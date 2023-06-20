from flask import Blueprint, render_template, request
from pymongo import MongoClient
import logging

from app.classifier import ClassiferSingleton

client = MongoClient('mongo:27017')
mongo_db = client['analytics']  
collection_analytics_v1 = mongo_db['analytics_v1']  

views = Blueprint('views', __name__,
                  template_folder='templates',
                  static_folder='static')


@views.route('/', methods=['GET', 'POST'])
def sentiment():
    link = False
    if request.method == "POST":
        link = request.form.get('videoQuery')
        ClassiferSingleton().make_analysis(link)
        link = link[link.rindex('?v=')+3:]
    return render_template(
        template_name_or_list='index.html',
        link=link
    )
    