from flask import Flask, request, redirect, url_for, session, abort, flash, jsonify, g
from flask import render_template
from flask import make_response
from middleware import *



def init_apiroutes(app):
    if app:
         app.add_url_rule('/home_page/<date>','home_page_date',home_page_date,methods=['GET','POST'])
        


def init_pageroutes(app):
    if app:
        app.add_url_rule('/','home_page',home_page,methods=['GET','POST'])
        