#!/bin/sh

SCRIPT_PATH=$(realpath "$0")
SCRIPTS_DIR=$(dirname $SCRIPT_PATH)
PROJECT_DIR=$(dirname $SCRIPTS_DIR)
SOURCES_DIR=$PROJECT_DIR/sources
SUBMIT_DIR=$PROJECT_DIR/submit
DOCS_DIR=$PROJECT_DIR/docs

rm $PROJECT_DIR/submit.tar.gz
rm -rf $SUBMIT_DIR

mkdir $SUBMIT_DIR
mkdir $SUBMIT_DIR/scripts

cp $SOURCES_DIR/template.cu $SUBMIT_DIR/template.cu
cp $DOCS_DIR/report.pdf $SUBMIT_DIR/report.pdf
cp $DOCS_DIR/evaluation.pdf $SUBMIT_DIR/evaluation.pdf
cp $SCRIPTS_DIR/manage.py $SUBMIT_DIR/scripts/manage.py
cp -r $SCRIPTS_DIR/templates $SUBMIT_DIR/scripts/templates

tar cvf submit.tar.gz submit