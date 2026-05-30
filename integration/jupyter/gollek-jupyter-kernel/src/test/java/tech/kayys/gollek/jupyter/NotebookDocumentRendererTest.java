package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class NotebookDocumentRendererTest {

    @Test
    void markdownPreviewRendersRichHtmlAndEscapesTarget() {
        NotebookPreview preview = NotebookDocumentRenderer.markdownPreview(
                "/tmp/<notes>.md",
                "# Title\n\n- <item>"
        );

        assertTrue(preview.plain().contains("Markdown(/tmp/<notes>.md)"));
        assertTrue(preview.plain().contains("- <item>"));
        assertTrue(preview.html().contains("/tmp/&lt;notes&gt;.md"));
        assertTrue(preview.html().contains("<h1>Title</h1>"));
        assertTrue(preview.html().contains("&lt;item&gt;"));
    }

    @Test
    void htmlPreviewWrapsRawHtmlButEscapesPath() {
        NotebookPreview preview = NotebookDocumentRenderer.htmlPreview(
                "/tmp/<page>.html",
                "<strong>Hello</strong>"
        );

        assertTrue(preview.plain().contains("HTML(/tmp/<page>.html)"));
        assertTrue(preview.html().contains("/tmp/&lt;page&gt;.html"));
        assertTrue(preview.html().contains("<strong>Hello</strong>"));
    }

    @Test
    void jsonPreviewEscapesTargetAndFormattedJson() {
        NotebookPreview preview = NotebookDocumentRenderer.jsonPreview(
                "/tmp/<data>.json",
                "{\n  \"name\": \"<ada>\"\n}"
        );

        assertTrue(preview.plain().contains("JSON(/tmp/<data>.json)"));
        assertTrue(preview.plain().contains("\"name\": \"<ada>\""));
        assertTrue(preview.html().contains("/tmp/&lt;data&gt;.json"));
        assertTrue(preview.html().contains("&quot;") || preview.html().contains("\"name\""));
        assertTrue(preview.html().contains("&lt;ada&gt;"));
        assertFalse(preview.html().contains("<ada>"));
    }

    @Test
    void syntaxPreviewMethodsEscapeTargetsAndHighlightContent() {
        NotebookPreview yaml = NotebookDocumentRenderer.yamlPreview("/tmp/<config>.yaml", "name: ada");
        NotebookPreview toml = NotebookDocumentRenderer.tomlPreview("/tmp/<config>.toml", "[user]\nname = \"ada\"");
        NotebookPreview xml = NotebookDocumentRenderer.xmlPreview("/tmp/<config>.xml", "<user name=\"ada\"/>");
        NotebookPreview ini = NotebookDocumentRenderer.iniPreview("/tmp/<config>.ini", "[user]\nname=ada");

        assertTrue(yaml.plain().contains("YAML(/tmp/<config>.yaml)"));
        assertTrue(yaml.html().contains("/tmp/&lt;config&gt;.yaml"));
        assertTrue(yaml.html().contains("<span style='color:#005cc5;font-weight:600'>name</span>"));
        assertTrue(toml.html().contains("<span style='color:#d73a49;font-weight:600'>[user]</span>"));
        assertTrue(xml.html().contains("/tmp/&lt;config&gt;.xml"));
        assertTrue(ini.html().contains("<span style='color:#d73a49;font-weight:600'>[user]</span>"));
    }

    @Test
    void propertiesPreviewSortsKeysAndEscapesValues() throws Exception {
        NotebookPreview preview = NotebookDocumentRenderer.propertiesPreview(
                "/tmp/<app>.properties",
                "z=last\na=<first>\n"
        );

        assertTrue(preview.plain().contains("Properties(/tmp/<app>.properties, entries=2)"));
        assertTrue(preview.plain().contains("a=<first>"));
        assertTrue(preview.plain().contains("z=last"));
        assertTrue(preview.plain().indexOf("a=<first>") < preview.plain().indexOf("z=last"));
        assertTrue(preview.html().contains("/tmp/&lt;app&gt;.properties"));
        assertTrue(preview.html().contains("&lt;first&gt;"));
        assertFalse(preview.html().contains("<first>"));
    }

    @Test
    void envFilePreviewParsesEntriesAndSkipsComments() {
        NotebookPreview preview = NotebookDocumentRenderer.envFilePreview(
                "/tmp/<app>.env",
                "# ignored\nUSER=ada\nTOKEN='<secret>'\ninvalid\n"
        );

        assertTrue(preview.plain().contains("EnvFile(/tmp/<app>.env, entries=2)"));
        assertTrue(preview.plain().contains("USER=ada"));
        assertTrue(preview.plain().contains("TOKEN=<secret>"));
        assertFalse(preview.plain().contains("ignored"));
        assertTrue(preview.html().contains("/tmp/&lt;app&gt;.env"));
        assertTrue(preview.html().contains("&lt;secret&gt;"));
    }
}
