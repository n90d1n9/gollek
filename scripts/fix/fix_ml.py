import os

build_path = './training/gollek-serializer/build.gradle.kts'
with open(build_path, 'r') as f:
    content = f.read()

deps = """
    implementation(project(":core:gollek-core"))
    implementation(project(":ml:gollek-ml-core"))
    implementation(project(":ml:gollek-ml-persistence"))
    implementation(project(":ml:gollek-ml-estimator"))
    implementation("com.google.code.gson:gson:2.10.1")
"""
content = content.replace('    implementation(project(":core:gollek-core"))', deps)
with open(build_path, 'w') as f:
    f.write(content)

pb_path = './training/gollek-serializer/src/main/java/tech/kayys/gollek/serializer/PythonBridge.java'
with open(pb_path, 'r') as f:
    pb_content = f.read()

pb_content = pb_content.replace('import tech.kayys.gollek.ml.ensemble.*;', 'import tech.kayys.gollek.train.estimator.*;')
with open(pb_path, 'w') as f:
    f.write(pb_content)

