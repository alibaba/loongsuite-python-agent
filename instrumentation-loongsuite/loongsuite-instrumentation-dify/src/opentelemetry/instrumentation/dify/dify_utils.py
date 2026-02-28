# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from opentelemetry.instrumentation.dify.config import (
    is_wrapper_version_1,
    is_wrapper_version_2,
)

try:
    from extensions.ext_database import db
    from models.model import App, Message

    dify_dependencies_available = True
except ImportError:
    dify_dependencies_available = False


def get_app_name_by_id(app_id: str) -> str:
    if not dify_dependencies_available:
        return app_id
    app_info = (
        db.session.query(
            App.id,
            App.name,
        )
        .filter(App.id == app_id)
        .all()
    )
    if len(app_info) <= 0:
        return app_id
    app_name = app_info[0].name
    if app_name is None:
        return app_id
    return app_name


def get_message_data(message_id: str):
    if not dify_dependencies_available:
        return None
    return db.session.query(Message).filter(Message.id == message_id).first()


def get_workflow_run_id(run):
    if is_wrapper_version_1():
        return getattr(run, "id", None)
    elif is_wrapper_version_2():
        return getattr(run, "id_", None)
    else:
        return None
