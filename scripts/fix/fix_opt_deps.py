import os

projects = [
    './optimization/gollek-plugin-perfmode/build.gradle.kts',
    './optimization/gollek-plugin-wait-scheduler/build.gradle.kts',
    './optimization/gollek-plugin-evicpress/build.gradle.kts',
    './optimization/gollek-plugin-prompt-cache/build.gradle.kts'
]

deps = """
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-plugin"))
    implementation(project(":core:plugin:gollek-plugin-optimization-core"))
"""

for p in projects:
    if os.path.exists(p):
        with open(p, 'r') as f:
            content = f.read()
        content = content.replace('dependencies {', 'dependencies {\n' + deps)
        with open(p, 'w') as f:
            f.write(content)
