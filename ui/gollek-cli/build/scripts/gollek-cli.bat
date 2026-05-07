@rem
@rem Copyright 2015 the original author or authors.
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem      https://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem
@rem SPDX-License-Identifier: Apache-2.0
@rem

@if "%DEBUG%"=="" @echo off
@rem ##########################################################################
@rem
@rem  gollek-cli startup script for Windows
@rem
@rem ##########################################################################

@rem Set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" setlocal

set DIRNAME=%~dp0
if "%DIRNAME%"=="" set DIRNAME=.
@rem This is normally unused
set APP_BASE_NAME=%~n0
set APP_HOME=%DIRNAME%..

@rem Resolve any "." and ".." in APP_HOME to make it shorter.
for %%i in ("%APP_HOME%") do set APP_HOME=%%~fi

@rem Add default JVM options here. You can also use JAVA_OPTS and GOLLEK_CLI_OPTS to pass JVM options to this script.
set DEFAULT_JVM_OPTS="--enable-native-access=ALL-UNNAMED" "--enable-preview"

@rem Find java.exe
if defined JAVA_HOME goto findJavaFromJavaHome

set JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
if %ERRORLEVEL% equ 0 goto execute

echo. 1>&2
echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH. 1>&2
echo. 1>&2
echo Please set the JAVA_HOME variable in your environment to match the 1>&2
echo location of your Java installation. 1>&2

goto fail

:findJavaFromJavaHome
set JAVA_HOME=%JAVA_HOME:"=%
set JAVA_EXE=%JAVA_HOME%/bin/java.exe

if exist "%JAVA_EXE%" goto execute

echo. 1>&2
echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME% 1>&2
echo. 1>&2
echo Please set the JAVA_HOME variable in your environment to match the 1>&2
echo location of your Java installation. 1>&2

goto fail

:execute
@rem Setup the command line

set CLASSPATH=%APP_HOME%\lib\gollek-cli-1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-sdk-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-sdk-core-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-sdk-session-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\log-parser-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-content-safety-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-semantic-cache-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-pii-redaction-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-mcp-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-rag-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-langchain4j-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-kernel-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-kernel.jar;%APP_HOME%\lib\gollek-plugin-runner-onnx-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-onnx-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-ml-export-onnx-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-runner-onnx-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-runner-litert-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-ml-litert-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-gguf-converter-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-gguf-converter-0.1.0-SNAPSHOT-jar-with-dependencies.jar;%APP_HOME%\lib\gollek-gguf-core-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-gguf-core-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-gguf-converter-java-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-gguf-core-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-safetensor-api-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-runner-safetensor-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-safetensor-spi-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-safetensor-core-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-safetensor-loader-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-safetensor-quantization-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-runner-safetensor-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-safetensor-rag-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-safetensor-engine-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-runner-tensorrt-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-ml-pickle-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-runner-gguf-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-openai-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-gemini-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-anthropic-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-cerebras-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-mistral-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-kernel-blackwell-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-kernel-blackwell-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-backend-metal-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-backend-metal-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-kernel-cuda-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-kernel-cuda-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-kernel-directml-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-plugin-kernel-rocm-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-kernel-rocm-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-model-repo-kaggle-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-model-repo-local-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-model-repo-hf-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-models-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\gollek-sdk-java-local-0.1.0-SNAPSHOT.jar;%APP_HOME%\lib\picocli-4.7.5.jar;%APP_HOME%\lib\gollek-spi-inference.jar;%APP_HOME%\lib\gollek-spi-model.jar;%APP_HOME%\lib\gollek-spi.jar;%APP_HOME%\lib\gollek-error-code.jar;%APP_HOME%\lib\jackson-annotations-2.16.1.jar;%APP_HOME%\lib\jackson-core-2.16.1.jar;%APP_HOME%\lib\gollek-spi-multimodal.jar;%APP_HOME%\lib\jackson-databind-2.16.1.jar;%APP_HOME%\lib\mutiny-2.5.5.jar;%APP_HOME%\lib\smallrye-common-annotation-2.2.0.jar;%APP_HOME%\lib\jakarta.validation-api-3.0.2.jar;%APP_HOME%\lib\commons-codec-1.16.1.jar;%APP_HOME%\lib\annotations-24.0.1.jar;%APP_HOME%\lib\jakarta.ws.rs-api-3.1.0.jar;%APP_HOME%\lib\gollek-ir.jar;%APP_HOME%\lib\gollek-tensor.jar;%APP_HOME%\lib\reactive-streams-1.0.4.jar;%APP_HOME%\lib\jboss-logging-3.5.3.Final.jar


@rem Execute gollek-cli
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %GOLLEK_CLI_OPTS%  -classpath "%CLASSPATH%" tech.kayys.gollek.cli.NewGollekCLI %*

:end
@rem End local scope for the variables with windows NT shell
if %ERRORLEVEL% equ 0 goto mainEnd

:fail
rem Set variable GOLLEK_CLI_EXIT_CONSOLE if you need the _script_ return code instead of
rem the _cmd.exe /c_ return code!
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% equ 0 set EXIT_CODE=1
if not ""=="%GOLLEK_CLI_EXIT_CONSOLE%" exit %EXIT_CODE%
exit /b %EXIT_CODE%

:mainEnd
if "%OS%"=="Windows_NT" endlocal

:omega
