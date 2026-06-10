import java.security.MessageDigest
import java.util.Properties
import org.gradle.api.provider.Provider
import org.gradle.api.tasks.PathSensitivity

plugins {
    java
    id("io.quarkus")
}

val quarkusVersion = rootProject.extra["quarkusVersion"] as String

data class ModelFamilyModule(val id: String, val path: String, val profile: String)

data class ModelFamilyBundleAlias(val id: String, val description: String, val familyIds: Set<String>)

data class ModelFamilyBundlePreset(
    val id: String,
    val description: String,
    val selectors: Set<String>,
    val requiredFamilies: Set<String> = emptySet(),
    val forbiddenFamilies: Set<String> = emptySet(),
    val requiredAliases: Set<String> = emptySet(),
    val forbiddenAliases: Set<String> = emptySet()
)

data class ModelFamilyBundleAliasCoverage(
    val alias: ModelFamilyBundleAlias,
    val selectedFamilyIds: Set<String>,
    val missingFamilyIds: Set<String>
) {
    val complete: Boolean = alias.familyIds.isNotEmpty() && missingFamilyIds.isEmpty()
    val partial: Boolean = !complete && selectedFamilyIds.isNotEmpty()
    fun compactSummary(): String {
        val status = when {
            complete -> "complete"
            partial -> "partial"
            else -> "empty"
        }
        return "${alias.id}(${selectedFamilyIds.size}/${alias.familyIds.size} $status)"
    }
}

val modelFamilyPluginDescriptorRelativePath = "src/main/resources/plugin.json"
val modelFamilyServiceDescriptorRelativePath =
    "src/main/resources/META-INF/services/tech.kayys.gollek.spi.model.ModelFamilyPlugin"
val gollekPluginServiceDescriptorRelativePath =
    "src/main/resources/META-INF/services/tech.kayys.gollek.spi.plugin.GollekPlugin"
val pendingModelFamilyTokenizerMetadataReasons: Map<String, String> by lazy {
    modelFamilyModules
        .filter { module -> modelFamilyProject(module) != null }
        .filter { module -> modelFamilyPluginDescriptorTokenizerMetadataPending(module.path) }
        .associate { module -> module.id to modelFamilyPluginDescriptorTokenizerMetadataPendingReason(module.path) }
}
val pendingModelFamilyTokenizerMetadata: Set<String> by lazy {
    pendingModelFamilyTokenizerMetadataReasons.keys
}

data class ModelFamilyBundlePresetValidation(
    val preset: ModelFamilyBundlePreset,
    val selectedFamilyIds: Set<String>,
    val productionTokenizerMetadataRequired: Boolean,
    val pendingTokenizerFamilyIds: Set<String>,
    val pendingTokenizerReasons: Map<String, String>,
    val configurationProblems: List<String>,
    val missingRequiredFamilyIds: Set<String>,
    val selectedForbiddenFamilyIds: Set<String>,
    val missingRequiredAliases: Map<String, Set<String>>,
    val selectedForbiddenAliases: Map<String, Set<String>>
) {
    val violationCount: Int
        get() = configurationProblems.size +
                missingRequiredFamilyIds.size +
                selectedForbiddenFamilyIds.size +
                missingRequiredAliases.size +
                selectedForbiddenAliases.size
    val passed: Boolean
        get() = violationCount == 0
    val productionTokenizerMetadataReady: Boolean
        get() = pendingTokenizerFamilyIds.isEmpty()
    val productionSafetyPassed: Boolean
        get() = !productionTokenizerMetadataRequired || productionTokenizerMetadataReady
    val productionSafetyViolationCount: Int
        get() = if (productionTokenizerMetadataRequired) pendingTokenizerFamilyIds.size else 0
}

data class ModelFamilyBundlePresetConformanceValidation(
    val presetId: String,
    val presetMetadataPresent: Boolean,
    val selectorsMatch: Boolean,
    val policyInputsMatch: Boolean,
    val explicitSelectorOverride: Boolean,
    val explicitPolicyOverride: Boolean,
    val selectorAdditions: Set<String>,
    val selectorOmissions: Set<String>,
    val requiredFamilyAdditions: Set<String>,
    val requiredFamilyOmissions: Set<String>,
    val forbiddenFamilyAdditions: Set<String>,
    val forbiddenFamilyOmissions: Set<String>,
    val requiredAliasAdditions: Set<String>,
    val requiredAliasOmissions: Set<String>,
    val forbiddenAliasAdditions: Set<String>,
    val forbiddenAliasOmissions: Set<String>
) {
    val hasPreset: Boolean
        get() = presetId.isNotBlank()
    val matchesPreset: Boolean
        get() = presetMetadataPresent && selectorsMatch && policyInputsMatch
    val cleanPresetBuild: Boolean
        get() = matchesPreset && !explicitSelectorOverride && !explicitPolicyOverride
    val statusLabel: String
        get() = when {
            !hasPreset -> "none"
            !presetMetadataPresent -> "unknown"
            cleanPresetBuild -> "clean"
            matchesPreset -> "matched"
            else -> "drifted"
        }
    val compactSummary: String
        get() = when {
            !hasPreset -> "none"
            !presetMetadataPresent -> "unknown (preset metadata missing)"
            cleanPresetBuild -> "clean"
            matchesPreset -> "matched with explicit overrides"
            else -> {
                val reasons = mutableListOf<String>()
                if (!selectorsMatch) {
                    reasons += "selectors differ"
                }
                if (!policyInputsMatch) {
                    reasons += "policy inputs differ"
                }
                if (explicitSelectorOverride) {
                    reasons += "explicit selector override"
                }
                if (explicitPolicyOverride) {
                    reasons += "explicit policy override"
                }
                "drifted (${reasons.joinToString("; ")})"
            }
        }
}

data class ModelFamilyBundleLockDriftEntry(
    val key: String,
    val expected: String,
    val actual: String?,
    val status: String,
    val conformanceRelated: Boolean,
    val fixtureRelated: Boolean
)

data class ModelFamilyBundleSelectionDriftEntry(
    val key: String,
    val scope: String,
    val value: String,
    val status: String
)

data class ModelFamilyBundleScalarDriftEntry(
    val key: String,
    val scope: String,
    val label: String,
    val status: String,
    val expected: String,
    val actual: String?
)

data class ModelFamilyFixtureFingerprintDriftEntry(
    val key: String,
    val scope: String,
    val familyId: String,
    val status: String,
    val expected: String?,
    val actual: String?
)

data class ModelFamilyFixtureInventory(
    val module: ModelFamilyModule,
    val selected: Boolean,
    val required: Boolean,
    val fixturePath: String,
    val present: Boolean,
    val configPresent: Boolean,
    val modelTypes: List<String>,
    val architectures: List<String>,
    val tokenizerFiles: List<String>,
    val contentFingerprint: String,
    val problems: List<String>
) {
    val passed: Boolean
        get() = problems.isEmpty()
}

data class ModelFamilyFixtureLockSnapshot(
    val requiredSelectors: List<String>,
    val requiredFamilies: List<String>,
    val passed: Boolean,
    val availableFamilyCount: Int,
    val fixtureFamilyCount: Int,
    val requiredFamilyCount: Int,
    val requiredPassedCount: Int,
    val missingRequiredCount: Int,
    val problemFamilyCount: Int,
    val missingRequiredFamilies: List<String>,
    val problemFamilies: List<String>,
    val requiredFingerprint: String,
    val inventoryFingerprint: String,
    val requiredContentFingerprints: Map<String, String>,
    val presentContentFingerprints: Map<String, String>
)

val modelFamilyModules = listOf(
    ModelFamilyModule("albert", ":models:gollek-model-albert", "metadata_only"),
    ModelFamilyModule("align", ":models:gollek-model-align", "metadata_only"),
    ModelFamilyModule("altclip", ":models:gollek-model-altclip", "metadata_only"),
    ModelFamilyModule("arcee", ":models:gollek-model-arcee", "metadata_only"),
    ModelFamilyModule("bart", ":models:gollek-model-bart", "metadata_only"),
    ModelFamilyModule("bamba", ":models:gollek-model-bamba", "metadata_only"),
    ModelFamilyModule("bark", ":models:gollek-model-bark", "metadata_only"),
    ModelFamilyModule("barthez", ":models:gollek-model-barthez", "metadata_only"),
    ModelFamilyModule("bartpho", ":models:gollek-model-bartpho", "metadata_only"),
    ModelFamilyModule("beit", ":models:gollek-model-beit", "metadata_only"),
    ModelFamilyModule("bert", ":models:gollek-model-bert", "metadata_only"),
    ModelFamilyModule("bert_generation", ":models:gollek-model-bert-generation", "metadata_only"),
    ModelFamilyModule("bert_japanese", ":models:gollek-model-bert-japanese", "metadata_only"),
    ModelFamilyModule("bertweet", ":models:gollek-model-bertweet", "metadata_only"),
    ModelFamilyModule("bigbird", ":models:gollek-model-bigbird", "metadata_only"),
    ModelFamilyModule("bigbird_pegasus", ":models:gollek-model-bigbird-pegasus", "metadata_only"),
    ModelFamilyModule("biogpt", ":models:gollek-model-biogpt", "metadata_only"),
    ModelFamilyModule("bitnet", ":models:gollek-model-bitnet", "metadata_only"),
    ModelFamilyModule("blenderbot", ":models:gollek-model-blenderbot", "metadata_only"),
    ModelFamilyModule("blenderbot_small", ":models:gollek-model-blenderbot-small", "metadata_only"),
    ModelFamilyModule("blip", ":models:gollek-model-blip", "metadata_only"),
    ModelFamilyModule("blip_2", ":models:gollek-model-blip-2", "metadata_only"),
    ModelFamilyModule("bloom", ":models:gollek-model-bloom", "metadata_only"),
    ModelFamilyModule("bridgetower", ":models:gollek-model-bridgetower", "metadata_only"),
    ModelFamilyModule("bros", ":models:gollek-model-bros", "metadata_only"),
    ModelFamilyModule("byt5", ":models:gollek-model-byt5", "metadata_only"),
    ModelFamilyModule("camembert", ":models:gollek-model-camembert", "metadata_only"),
    ModelFamilyModule("canine", ":models:gollek-model-canine", "metadata_only"),
    ModelFamilyModule("chameleon", ":models:gollek-model-chameleon", "metadata_only"),
    ModelFamilyModule("chinese_clip", ":models:gollek-model-chinese-clip", "metadata_only"),
    ModelFamilyModule("clap", ":models:gollek-model-clap", "metadata_only"),
    ModelFamilyModule("clip", ":models:gollek-model-clip", "metadata_only"),
    ModelFamilyModule("clipseg", ":models:gollek-model-clipseg", "metadata_only"),
    ModelFamilyModule("code_llama", ":models:gollek-model-code-llama", "optional"),
    ModelFamilyModule("codegen", ":models:gollek-model-codegen", "metadata_only"),
    ModelFamilyModule("cohere", ":models:gollek-model-cohere", "optional"),
    ModelFamilyModule("colmodernvbert", ":models:gollek-model-colmodernvbert", "metadata_only"),
    ModelFamilyModule("colpali", ":models:gollek-model-colpali", "metadata_only"),
    ModelFamilyModule("colqwen2", ":models:gollek-model-colqwen2", "metadata_only"),
    ModelFamilyModule("convbert", ":models:gollek-model-convbert", "metadata_only"),
    ModelFamilyModule("convnext", ":models:gollek-model-convnext", "metadata_only"),
    ModelFamilyModule("cpm", ":models:gollek-model-cpm", "metadata_only"),
    ModelFamilyModule("cpmant", ":models:gollek-model-cpmant", "metadata_only"),
    ModelFamilyModule("ctrl", ":models:gollek-model-ctrl", "metadata_only"),
    ModelFamilyModule("cvt", ":models:gollek-model-cvt", "metadata_only"),
    ModelFamilyModule("data2vec", ":models:gollek-model-data2vec", "metadata_only"),
    ModelFamilyModule("dbrx", ":models:gollek-model-dbrx", "metadata_only"),
    ModelFamilyModule("deberta", ":models:gollek-model-deberta", "metadata_only"),
    ModelFamilyModule("deberta_v2", ":models:gollek-model-deberta-v2", "metadata_only"),
    ModelFamilyModule("deepseek", ":models:gollek-model-deepseek", "experimental"),
    ModelFamilyModule("deepseek_v2", ":models:gollek-model-deepseek-v2", "experimental"),
    ModelFamilyModule("deepseek_v3", ":models:gollek-model-deepseek-v3", "experimental"),
    ModelFamilyModule("deepseek_vl", ":models:gollek-model-deepseek-vl", "experimental"),
    ModelFamilyModule("deepseek_vl_hybrid", ":models:gollek-model-deepseek-vl-hybrid", "experimental"),
    ModelFamilyModule("depth", ":models:gollek-model-depth", "metadata_only"),
    ModelFamilyModule("deit", ":models:gollek-model-deit", "metadata_only"),
    ModelFamilyModule("detr", ":models:gollek-model-detr", "metadata_only"),
    ModelFamilyModule("dialogpt", ":models:gollek-model-dialogpt", "metadata_only"),
    ModelFamilyModule("diffllama", ":models:gollek-model-diffllama", "metadata_only"),
    ModelFamilyModule("dinov2", ":models:gollek-model-dinov2", "metadata_only"),
    ModelFamilyModule("dinov2_with_registers", ":models:gollek-model-dinov2-with-registers", "metadata_only"),
    ModelFamilyModule("distilbert", ":models:gollek-model-distilbert", "metadata_only"),
    ModelFamilyModule("donut", ":models:gollek-model-donut", "metadata_only"),
    ModelFamilyModule("dpr", ":models:gollek-model-dpr", "metadata_only"),
    ModelFamilyModule("electra", ":models:gollek-model-electra", "metadata_only"),
    ModelFamilyModule("encodec", ":models:gollek-model-encodec", "metadata_only"),
    ModelFamilyModule("efficientnet", ":models:gollek-model-efficientnet", "metadata_only"),
    ModelFamilyModule("ernie", ":models:gollek-model-ernie", "metadata_only"),
    ModelFamilyModule("esm", ":models:gollek-model-esm", "metadata_only"),
    ModelFamilyModule("exaone", ":models:gollek-model-exaone", "metadata_only"),
    ModelFamilyModule("eurobert", ":models:gollek-model-eurobert", "metadata_only"),
    ModelFamilyModule("falcon", ":models:gollek-model-falcon", "metadata_only"),
    ModelFamilyModule("falcon_h1", ":models:gollek-model-falcon-h1", "metadata_only"),
    ModelFamilyModule("falcon_mamba", ":models:gollek-model-falcon-mamba", "metadata_only"),
    ModelFamilyModule("fast_vlm", ":models:gollek-model-fast-vlm", "metadata_only"),
    ModelFamilyModule("flava", ":models:gollek-model-flava", "metadata_only"),
    ModelFamilyModule("flaubert", ":models:gollek-model-flaubert", "metadata_only"),
    ModelFamilyModule("florence2", ":models:gollek-model-florence2", "metadata_only"),
    ModelFamilyModule("focalnet", ":models:gollek-model-focalnet", "metadata_only"),
    ModelFamilyModule("fnet", ":models:gollek-model-fnet", "metadata_only"),
    ModelFamilyModule("fsmt", ":models:gollek-model-fsmt", "metadata_only"),
    ModelFamilyModule("funnel", ":models:gollek-model-funnel", "metadata_only"),
    ModelFamilyModule("fuyu", ":models:gollek-model-fuyu", "metadata_only"),
    ModelFamilyModule("gemma", ":models:gollek-model-gemma", "core"),
    ModelFamilyModule("gemma2", ":models:gollek-model-gemma2", "optional"),
    ModelFamilyModule("gemma3", ":models:gollek-model-gemma3", "optional"),
    ModelFamilyModule("gemma3n", ":models:gollek-model-gemma3n", "metadata_only"),
    ModelFamilyModule("gemma4", ":models:gollek-model-gemma4", "optional"),
    ModelFamilyModule("glm", ":models:gollek-model-glm", "metadata_only"),
    ModelFamilyModule("glm_ocr", ":models:gollek-model-glm-ocr", "metadata_only"),
    ModelFamilyModule("got_ocr2", ":models:gollek-model-got-ocr2", "metadata_only"),
    ModelFamilyModule("gpt2", ":models:gollek-model-gpt2", "metadata_only"),
    ModelFamilyModule("gpt_bigcode", ":models:gollek-model-gpt-bigcode", "metadata_only"),
    ModelFamilyModule("gpt_neo", ":models:gollek-model-gpt-neo", "metadata_only"),
    ModelFamilyModule("gpt_neox", ":models:gollek-model-gpt-neox", "metadata_only"),
    ModelFamilyModule("gpt_neox_japanese", ":models:gollek-model-gpt-neox-japanese", "metadata_only"),
    ModelFamilyModule("gpt_oss", ":models:gollek-model-gpt-oss", "metadata_only"),
    ModelFamilyModule("gptj", ":models:gollek-model-gptj", "metadata_only"),
    ModelFamilyModule("granite", ":models:gollek-model-granite", "metadata_only"),
    ModelFamilyModule("grounding_dino", ":models:gollek-model-grounding-dino", "metadata_only"),
    ModelFamilyModule("groupvit", ":models:gollek-model-groupvit", "metadata_only"),
    ModelFamilyModule("herbert", ":models:gollek-model-herbert", "metadata_only"),
    ModelFamilyModule("hubert", ":models:gollek-model-hubert", "metadata_only"),
    ModelFamilyModule("ibert", ":models:gollek-model-ibert", "metadata_only"),
    ModelFamilyModule("idefics", ":models:gollek-model-idefics", "metadata_only"),
    ModelFamilyModule("idefics2", ":models:gollek-model-idefics2", "metadata_only"),
    ModelFamilyModule("idefics3", ":models:gollek-model-idefics3", "metadata_only"),
    ModelFamilyModule("instructblip", ":models:gollek-model-instructblip", "metadata_only"),
    ModelFamilyModule("instructblipvideo", ":models:gollek-model-instructblipvideo", "metadata_only"),
    ModelFamilyModule("internvl", ":models:gollek-model-internvl", "metadata_only"),
    ModelFamilyModule("jamba", ":models:gollek-model-jamba", "metadata_only"),
    ModelFamilyModule("jina_embeddings_v3", ":models:gollek-model-jina-embeddings-v3", "metadata_only"),
    ModelFamilyModule("kimi", ":models:gollek-model-kimi", "experimental"),
    ModelFamilyModule("kosmos2", ":models:gollek-model-kosmos2", "metadata_only"),
    ModelFamilyModule("kosmos2_5", ":models:gollek-model-kosmos2-5", "metadata_only"),
    ModelFamilyModule("layoutlm", ":models:gollek-model-layoutlm", "metadata_only"),
    ModelFamilyModule("layoutxlm", ":models:gollek-model-layoutxlm", "metadata_only"),
    ModelFamilyModule("levit", ":models:gollek-model-levit", "metadata_only"),
    ModelFamilyModule("lighton_ocr", ":models:gollek-model-lighton-ocr", "metadata_only"),
    ModelFamilyModule("lilt", ":models:gollek-model-lilt", "metadata_only"),
    ModelFamilyModule("llama", ":models:gollek-model-llama", "core"),
    ModelFamilyModule("llama4", ":models:gollek-model-llama4", "metadata_only"),
    ModelFamilyModule("llava", ":models:gollek-model-llava", "metadata_only"),
    ModelFamilyModule("llava_next", ":models:gollek-model-llava-next", "metadata_only"),
    ModelFamilyModule("llava_next_video", ":models:gollek-model-llava-next-video", "metadata_only"),
    ModelFamilyModule("llava_onevision", ":models:gollek-model-llava-onevision", "metadata_only"),
    ModelFamilyModule("longformer", ":models:gollek-model-longformer", "metadata_only"),
    ModelFamilyModule("longt5", ":models:gollek-model-longt5", "metadata_only"),
    ModelFamilyModule("luke", ":models:gollek-model-luke", "metadata_only"),
    ModelFamilyModule("lxmert", ":models:gollek-model-lxmert", "metadata_only"),
    ModelFamilyModule("m2m100", ":models:gollek-model-m2m100", "metadata_only"),
    ModelFamilyModule("mamba", ":models:gollek-model-mamba", "metadata_only"),
    ModelFamilyModule("mamba2", ":models:gollek-model-mamba2", "metadata_only"),
    ModelFamilyModule("marian", ":models:gollek-model-marian", "metadata_only"),
    ModelFamilyModule("markuplm", ":models:gollek-model-markuplm", "metadata_only"),
    ModelFamilyModule("maskformer", ":models:gollek-model-maskformer", "metadata_only"),
    ModelFamilyModule("mbart", ":models:gollek-model-mbart", "metadata_only"),
    ModelFamilyModule("megatron_bert", ":models:gollek-model-megatron-bert", "metadata_only"),
    ModelFamilyModule("megatron_gpt2", ":models:gollek-model-megatron-gpt2", "metadata_only"),
    ModelFamilyModule("metaclip_2", ":models:gollek-model-metaclip-2", "metadata_only"),
    ModelFamilyModule("mgp_str", ":models:gollek-model-mgp-str", "metadata_only"),
    ModelFamilyModule("ministral", ":models:gollek-model-ministral", "metadata_only"),
    ModelFamilyModule("ministral3", ":models:gollek-model-ministral3", "metadata_only"),
    ModelFamilyModule("mistral", ":models:gollek-model-mistral", "core"),
    ModelFamilyModule("mistral3", ":models:gollek-model-mistral3", "metadata_only"),
    ModelFamilyModule("mistral4", ":models:gollek-model-mistral4", "metadata_only"),
    ModelFamilyModule("mixtral", ":models:gollek-model-mixtral", "metadata_only"),
    ModelFamilyModule("mllama", ":models:gollek-model-mllama", "metadata_only"),
    ModelFamilyModule("mobilebert", ":models:gollek-model-mobilebert", "metadata_only"),
    ModelFamilyModule("mobilenet", ":models:gollek-model-mobilenet", "metadata_only"),
    ModelFamilyModule("mobilevit", ":models:gollek-model-mobilevit", "metadata_only"),
    ModelFamilyModule("mobilevitv2", ":models:gollek-model-mobilevitv2", "metadata_only"),
    ModelFamilyModule("modernbert", ":models:gollek-model-modernbert", "metadata_only"),
    ModelFamilyModule("modernbert_decoder", ":models:gollek-model-modernbert-decoder", "metadata_only"),
    ModelFamilyModule("modernvbert", ":models:gollek-model-modernvbert", "metadata_only"),
    ModelFamilyModule("mpt", ":models:gollek-model-mpt", "metadata_only"),
    ModelFamilyModule("mpnet", ":models:gollek-model-mpnet", "metadata_only"),
    ModelFamilyModule("mt5", ":models:gollek-model-mt5", "metadata_only"),
    ModelFamilyModule("musicgen", ":models:gollek-model-musicgen", "metadata_only"),
    ModelFamilyModule("nemotron", ":models:gollek-model-nemotron", "metadata_only"),
    ModelFamilyModule("nllb", ":models:gollek-model-nllb", "metadata_only"),
    ModelFamilyModule("nllb_moe", ":models:gollek-model-nllb-moe", "metadata_only"),
    ModelFamilyModule("nomic_bert", ":models:gollek-model-nomic-bert", "metadata_only"),
    ModelFamilyModule("nougat", ":models:gollek-model-nougat", "metadata_only"),
    ModelFamilyModule("olmo", ":models:gollek-model-olmo", "metadata_only"),
    ModelFamilyModule("olmo2", ":models:gollek-model-olmo2", "metadata_only"),
    ModelFamilyModule("olmo3", ":models:gollek-model-olmo3", "metadata_only"),
    ModelFamilyModule("olmo_hybrid", ":models:gollek-model-olmo-hybrid", "metadata_only"),
    ModelFamilyModule("olmoe", ":models:gollek-model-olmoe", "metadata_only"),
    ModelFamilyModule("oneformer", ":models:gollek-model-oneformer", "metadata_only"),
    ModelFamilyModule("openai_gpt", ":models:gollek-model-openai-gpt", "metadata_only"),
    ModelFamilyModule("opt", ":models:gollek-model-opt", "metadata_only"),
    ModelFamilyModule("owlvit", ":models:gollek-model-owlvit", "metadata_only"),
    ModelFamilyModule("owlv2", ":models:gollek-model-owlv2", "metadata_only"),
    ModelFamilyModule("paligemma", ":models:gollek-model-paligemma", "metadata_only"),
    ModelFamilyModule("pegasus", ":models:gollek-model-pegasus", "metadata_only"),
    ModelFamilyModule("pegasus_x", ":models:gollek-model-pegasus-x", "metadata_only"),
    ModelFamilyModule("persimmon", ":models:gollek-model-persimmon", "metadata_only"),
    ModelFamilyModule("phi", ":models:gollek-model-phi", "core"),
    ModelFamilyModule("phi4_multimodal", ":models:gollek-model-phi4-multimodal", "metadata_only"),
    ModelFamilyModule("phobert", ":models:gollek-model-phobert", "metadata_only"),
    ModelFamilyModule("phimoe", ":models:gollek-model-phimoe", "metadata_only"),
    ModelFamilyModule("pix2struct", ":models:gollek-model-pix2struct", "metadata_only"),
    ModelFamilyModule("pixtral", ":models:gollek-model-pixtral", "metadata_only"),
    ModelFamilyModule("plbart", ":models:gollek-model-plbart", "metadata_only"),
    ModelFamilyModule("poolformer", ":models:gollek-model-poolformer", "metadata_only"),
    ModelFamilyModule("prophetnet", ":models:gollek-model-prophetnet", "metadata_only"),
    ModelFamilyModule("pvt", ":models:gollek-model-pvt", "metadata_only"),
    ModelFamilyModule("pvt_v2", ":models:gollek-model-pvt-v2", "metadata_only"),
    ModelFamilyModule("qwen", ":models:gollek-model-qwen", "core"),
    ModelFamilyModule("qwen2_audio", ":models:gollek-model-qwen2-audio", "metadata_only"),
    ModelFamilyModule("qwen2_moe", ":models:gollek-model-qwen2-moe", "metadata_only"),
    ModelFamilyModule("qwen2_5_omni", ":models:gollek-model-qwen2-5-omni", "metadata_only"),
    ModelFamilyModule("qwen2_vl", ":models:gollek-model-qwen2-vl", "metadata_only"),
    ModelFamilyModule("qwen2_5_vl", ":models:gollek-model-qwen2-5-vl", "metadata_only"),
    ModelFamilyModule("qwen3_5", ":models:gollek-model-qwen3-5", "metadata_only"),
    ModelFamilyModule("qwen3_5_moe", ":models:gollek-model-qwen3-5-moe", "metadata_only"),
    ModelFamilyModule("qwen3_moe", ":models:gollek-model-qwen3-moe", "metadata_only"),
    ModelFamilyModule("qwen3_next", ":models:gollek-model-qwen3-next", "metadata_only"),
    ModelFamilyModule("qwen3_omni_moe", ":models:gollek-model-qwen3-omni-moe", "metadata_only"),
    ModelFamilyModule("qwen3_vl", ":models:gollek-model-qwen3-vl", "metadata_only"),
    ModelFamilyModule("qwen3_vl_moe", ":models:gollek-model-qwen3-vl-moe", "metadata_only"),
    ModelFamilyModule("qwen_vl", ":models:gollek-model-qwen-vl", "metadata_only"),
    ModelFamilyModule("rag", ":models:gollek-model-rag", "metadata_only"),
    ModelFamilyModule("recurrent_gemma", ":models:gollek-model-recurrent-gemma", "metadata_only"),
    ModelFamilyModule("reformer", ":models:gollek-model-reformer", "metadata_only"),
    ModelFamilyModule("regnet", ":models:gollek-model-regnet", "metadata_only"),
    ModelFamilyModule("rembert", ":models:gollek-model-rembert", "metadata_only"),
    ModelFamilyModule("resnet", ":models:gollek-model-resnet", "metadata_only"),
    ModelFamilyModule("roberta", ":models:gollek-model-roberta", "metadata_only"),
    ModelFamilyModule("roberta_prelayernorm", ":models:gollek-model-roberta-prelayernorm", "metadata_only"),
    ModelFamilyModule("roc_bert", ":models:gollek-model-roc-bert", "metadata_only"),
    ModelFamilyModule("roformer", ":models:gollek-model-roformer", "metadata_only"),
    ModelFamilyModule("rwkv", ":models:gollek-model-rwkv", "metadata_only"),
    ModelFamilyModule("sam", ":models:gollek-model-sam", "metadata_only"),
    ModelFamilyModule("segformer", ":models:gollek-model-segformer", "metadata_only"),
    ModelFamilyModule("seamless_m4t", ":models:gollek-model-seamless-m4t", "metadata_only"),
    ModelFamilyModule("shieldgemma2", ":models:gollek-model-shieldgemma2", "metadata_only"),
    ModelFamilyModule("siglip", ":models:gollek-model-siglip", "metadata_only"),
    ModelFamilyModule("siglip2", ":models:gollek-model-siglip2", "metadata_only"),
    ModelFamilyModule("smollm3", ":models:gollek-model-smollm3", "metadata_only"),
    ModelFamilyModule("smolvlm", ":models:gollek-model-smolvlm", "metadata_only"),
    ModelFamilyModule("speecht5", ":models:gollek-model-speecht5", "metadata_only"),
    ModelFamilyModule("squeezebert", ":models:gollek-model-squeezebert", "metadata_only"),
    ModelFamilyModule("stablelm", ":models:gollek-model-stablelm", "metadata_only"),
    ModelFamilyModule("starcoder2", ":models:gollek-model-starcoder2", "metadata_only"),
    ModelFamilyModule("switch_transformers", ":models:gollek-model-switch-transformers", "metadata_only"),
    ModelFamilyModule("swin", ":models:gollek-model-swin", "metadata_only"),
    ModelFamilyModule("t5", ":models:gollek-model-t5", "metadata_only"),
    ModelFamilyModule("t5gemma", ":models:gollek-model-t5gemma", "metadata_only"),
    ModelFamilyModule("t5gemma2", ":models:gollek-model-t5gemma2", "metadata_only"),
    ModelFamilyModule("trocr", ":models:gollek-model-trocr", "metadata_only"),
    ModelFamilyModule("umt5", ":models:gollek-model-umt5", "metadata_only"),
    ModelFamilyModule("upernet", ":models:gollek-model-upernet", "metadata_only"),
    ModelFamilyModule("vaultgemma", ":models:gollek-model-vaultgemma", "metadata_only"),
    ModelFamilyModule("video_llama_3", ":models:gollek-model-video-llama-3", "metadata_only"),
    ModelFamilyModule("video_llava", ":models:gollek-model-video-llava", "metadata_only"),
    ModelFamilyModule("vipllava", ":models:gollek-model-vipllava", "metadata_only"),
    ModelFamilyModule("vilt", ":models:gollek-model-vilt", "metadata_only"),
    ModelFamilyModule("vision_text_dual_encoder", ":models:gollek-model-vision-text-dual-encoder", "metadata_only"),
    ModelFamilyModule("vit", ":models:gollek-model-vit", "metadata_only"),
    ModelFamilyModule("visual_bert", ":models:gollek-model-visual-bert", "metadata_only"),
    ModelFamilyModule("vits", ":models:gollek-model-vits", "metadata_only"),
    ModelFamilyModule("wav2vec2", ":models:gollek-model-wav2vec2", "metadata_only"),
    ModelFamilyModule("wav2vec2_bert", ":models:gollek-model-wav2vec2-bert", "metadata_only"),
    ModelFamilyModule("wavlm", ":models:gollek-model-wavlm", "metadata_only"),
    ModelFamilyModule("whisper", ":models:gollek-model-whisper", "metadata_only"),
    ModelFamilyModule("x_clip", ":models:gollek-model-x-clip", "metadata_only"),
    ModelFamilyModule("xglm", ":models:gollek-model-xglm", "metadata_only"),
    ModelFamilyModule("xlm", ":models:gollek-model-xlm", "metadata_only"),
    ModelFamilyModule("xlm_roberta", ":models:gollek-model-xlm-roberta", "metadata_only"),
    ModelFamilyModule("xlm_roberta_xl", ":models:gollek-model-xlm-roberta-xl", "metadata_only"),
    ModelFamilyModule("xlnet", ":models:gollek-model-xlnet", "metadata_only"),
    ModelFamilyModule("xmod", ":models:gollek-model-xmod", "metadata_only"),
    ModelFamilyModule("yolos", ":models:gollek-model-yolos", "metadata_only"),
    ModelFamilyModule("yi", ":models:gollek-model-yii", "optional"),
    ModelFamilyModule("zamba", ":models:gollek-model-zamba", "metadata_only"),
)

val modelFamilyIds = modelFamilyModules.map { it.id }.toSet()

val duplicateModelFamilyModuleIds = modelFamilyModules
    .groupingBy { it.id }
    .eachCount()
    .filterValues { it > 1 }
    .keys
    .sorted()
if (duplicateModelFamilyModuleIds.isNotEmpty()) {
    throw GradleException(
        "Model-family modules must have unique IDs: ${duplicateModelFamilyModuleIds.joinToString(", ")}"
    )
}

val duplicateModelFamilyModulePaths = modelFamilyModules
    .groupingBy { it.path }
    .eachCount()
    .filterValues { it > 1 }
    .keys
    .sorted()
if (duplicateModelFamilyModulePaths.isNotEmpty()) {
    throw GradleException(
        "Model-family modules must have unique Gradle paths: ${duplicateModelFamilyModulePaths.joinToString(", ")}"
    )
}

val knownModelFamilyProfiles = setOf("core", "optional", "metadata_only", "experimental")
val unknownModelFamilyProfiles = modelFamilyModules
    .filter { it.profile !in knownModelFamilyProfiles }
    .map { "${it.id}:${it.profile}" }
    .sorted()
if (unknownModelFamilyProfiles.isNotEmpty()) {
    throw GradleException(
        "Model-family modules reference unknown profiles: ${unknownModelFamilyProfiles.joinToString(", ")}"
    )
}

fun familySet(vararg ids: String): Set<String> {
    return ids.toSet()
}

fun familySetByProfile(vararg profiles: String): Set<String> {
    val selectedProfiles = profiles.toSet()
    return modelFamilyModules
        .filter { module -> module.profile in selectedProfiles }
        .map { it.id }
        .toSet()
}

val modelFamilyBundleAliases = listOf(
    ModelFamilyBundleAlias(
        "direct",
        "Families with direct SafeTensor architecture adapters or guarded direct-runtime paths.",
        familySet(
            "code_llama", "cohere", "gemma", "gemma2", "gemma3", "gemma4",
            "llama", "mistral", "phi", "qwen", "yi"
        )
    ),
    ModelFamilyBundleAlias(
        "vlm",
        "Vision-language and multimodal generation families that should stay opt-in for production.",
        familySet(
            "blip", "blip_2", "chameleon", "fast_vlm", "florence2", "fuyu",
            "gemma3n", "gemma4", "glm_ocr", "got_ocr2", "idefics", "idefics2",
            "idefics3", "instructblip", "instructblipvideo", "internvl", "kosmos2",
            "kosmos2_5", "llama4", "llava", "llava_next", "llava_next_video",
            "llava_onevision", "mllama", "nougat", "paligemma", "phi4_multimodal",
            "pix2struct", "pixtral", "qwen_vl", "qwen2_vl", "qwen2_5_vl",
            "qwen3_vl", "qwen3_vl_moe", "smolvlm", "t5gemma2", "video_llama_3",
            "video_llava", "vipllava", "vilt", "visual_bert"
        )
    ),
    ModelFamilyBundleAlias(
        "vision",
        "Image, detection, document-vision, and vision-language families.",
        familySet(
            "align", "altclip", "beit", "blip", "blip_2", "bridgetower", "chameleon",
            "chinese_clip", "clip", "clipseg", "colpali", "colqwen2", "convnext",
            "cvt", "deit", "depth", "detr", "dinov2", "dinov2_with_registers",
            "donut", "efficientnet", "fast_vlm", "flava", "florence2", "focalnet",
            "fuyu", "gemma3n", "gemma4", "glm_ocr", "got_ocr2", "grounding_dino",
            "groupvit", "idefics", "idefics2", "idefics3", "instructblip",
            "instructblipvideo", "internvl", "kosmos2", "kosmos2_5", "layoutlm",
            "layoutxlm", "levit", "lighton_ocr", "llama4", "llava", "llava_next",
            "llava_next_video", "llava_onevision", "lxmert", "metaclip_2", "mllama",
            "mobilenet", "mobilevit", "mobilevitv2", "nougat", "oneformer", "owlv2",
            "owlvit", "paligemma", "phi4_multimodal", "pix2struct", "pixtral",
            "poolformer", "pvt", "pvt_v2", "qwen_vl", "qwen2_vl", "qwen2_5_vl",
            "qwen3_vl", "qwen3_vl_moe", "regnet", "resnet", "sam", "segformer",
            "shieldgemma2", "siglip", "siglip2", "smolvlm", "swin", "t5gemma2",
            "trocr", "upernet", "video_llama_3", "video_llava", "vipllava", "vilt",
            "vision_text_dual_encoder", "visual_bert", "vit", "x_clip", "yolos"
        )
    ),
    ModelFamilyBundleAlias(
        "audio",
        "Speech, audio-codec, audio-language, and omni multimodal families.",
        familySet(
            "bark", "clap", "encodec", "gemma3n", "gemma4", "hubert", "musicgen",
            "phi4_multimodal", "qwen2_audio", "qwen2_5_omni", "qwen3_omni_moe",
            "seamless_m4t", "speecht5", "vits", "wav2vec2", "wav2vec2_bert",
            "wavlm", "whisper"
        )
    ),
    ModelFamilyBundleAlias(
        "embedding",
        "Encoder, retrieval, and embedding-oriented families.",
        familySet(
            "albert", "bert", "bert_japanese", "bertweet", "camembert", "canine",
            "colmodernvbert", "colpali", "colqwen2", "convbert", "data2vec",
            "deberta", "deberta_v2", "distilbert", "dpr", "electra", "ernie", "esm",
            "eurobert", "flaubert", "fnet", "funnel", "herbert", "ibert",
            "jina_embeddings_v3", "layoutlm", "layoutxlm", "lilt", "longformer",
            "luke", "megatron_bert", "mobilebert", "modernbert", "modernvbert",
            "mpnet", "nomic_bert", "phobert", "rembert", "roberta",
            "roberta_prelayernorm", "roc_bert", "roformer", "squeezebert",
            "wav2vec2_bert", "xlm", "xlm_roberta", "xlm_roberta_xl", "xmod"
        )
    ),
    ModelFamilyBundleAlias(
        "moe",
        "Sparse expert, hybrid, and router-heavy families that should be explicitly opted into.",
        familySet(
            "bamba", "dbrx", "deepseek", "deepseek_v2", "deepseek_v3", "exaone",
            "gemma4", "granite", "jamba", "mixtral", "nllb_moe", "olmoe", "phimoe",
            "qwen2_moe", "qwen3_moe", "qwen3_5_moe", "qwen3_next",
            "qwen3_omni_moe", "qwen3_vl_moe", "switch_transformers"
        )
    ),
    ModelFamilyBundleAlias(
        "research",
        "All non-core model-family plugins for broad diagnostics and conversion planning.",
        familySetByProfile("optional", "metadata_only", "experimental")
    )
)

val unknownModelFamilyBundleAliasMembers = modelFamilyBundleAliases
    .flatMap { alias -> alias.familyIds.filter { familyId -> familyId !in modelFamilyIds }.map { alias.id to it } }
if (unknownModelFamilyBundleAliasMembers.isNotEmpty()) {
    throw GradleException(
        "Model-family bundle alias references unknown families: " +
                unknownModelFamilyBundleAliasMembers.joinToString(", ") { (alias, familyId) -> "$alias:$familyId" }
    )
}

val duplicateModelFamilyBundleAliasIds = modelFamilyBundleAliases
    .groupingBy { it.id }
    .eachCount()
    .filterValues { it > 1 }
    .keys
    .sorted()
if (duplicateModelFamilyBundleAliasIds.isNotEmpty()) {
    throw GradleException(
        "Model-family bundle aliases must have unique IDs: ${duplicateModelFamilyBundleAliasIds.joinToString(", ")}"
    )
}

val modelFamilyBundleAliasMap = modelFamilyBundleAliases.associateBy { it.id }
val modelFamilyBundleAliasIds = modelFamilyBundleAliasMap.keys

val modelFamilyBundlePresets = listOf(
    ModelFamilyBundlePreset(
        "prod_llm",
        "Lean production LLM server with direct-runtime families and embedding metadata detached.",
        selectors = setOf("direct"),
        requiredAliases = setOf("direct"),
        forbiddenAliases = setOf("embedding")
    ),
    ModelFamilyBundlePreset(
        "prod_embedding",
        "Retrieval or embedding service with encoder families attached and direct LLM families detached.",
        selectors = setOf("embedding"),
        requiredAliases = setOf("embedding"),
        forbiddenAliases = setOf("direct", "moe")
    ),
    ModelFamilyBundlePreset(
        "prod_vlm_metadata",
        "Production text stack plus VLM metadata for routing and conversion planning.",
        selectors = setOf("core", "vlm"),
        requiredFamilies = setOf("gemma", "llama", "mistral", "phi", "qwen"),
        requiredAliases = setOf("vlm"),
        forbiddenAliases = setOf("embedding")
    ),
    ModelFamilyBundlePreset(
        "metadata_inspection",
        "Metadata-only model farm for architecture/tokenizer inspection without direct-runtime families.",
        selectors = setOf("metadata_only"),
        forbiddenAliases = setOf("direct")
    ),
    ModelFamilyBundlePreset(
        "research_all",
        "Attach every model-family plugin for broad diagnostics and release validation.",
        selectors = setOf("all")
    ),
    ModelFamilyBundlePreset(
        "runtime_proxy",
        "Detach model-family plugins for a runtime-only proxy build.",
        selectors = setOf("none")
    )
)
val duplicateModelFamilyBundlePresetIds = modelFamilyBundlePresets
    .groupingBy { it.id }
    .eachCount()
    .filterValues { it > 1 }
    .keys
    .sorted()
if (duplicateModelFamilyBundlePresetIds.isNotEmpty()) {
    throw GradleException(
        "Model-family bundle presets must have unique IDs: ${duplicateModelFamilyBundlePresetIds.joinToString(", ")}"
    )
}

val modelFamilyBundlePresetMap = modelFamilyBundlePresets.associateBy { it.id }
val modelFamilyBundlePresetIds = modelFamilyBundlePresetMap.keys

fun expandModelFamilyBundleAliases(tokens: Set<String>): Set<String> {
    return tokens + tokens.flatMap { token -> modelFamilyBundleAliasMap[token]?.familyIds ?: emptySet() }
}

val availableModelFamilySelectors = (
        modelFamilyModules.map { it.id } +
                modelFamilyModules.map { it.profile } +
                modelFamilyBundleAliasIds +
                listOf("all", "none")
        ).distinct().sorted()

fun parseModelFamilySelectorTokens(value: String, emptyDefault: Set<String>): Set<String> {
    val tokens = value.split(",")
        .map { it.trim().lowercase().replace("-", "_") }
        .filter { it.isNotBlank() }
        .toSet()
    return tokens.ifEmpty { emptyDefault }
}

val requestedModelFamilyBundlePresetId = providers.gradleProperty("gollek.modelFamilyBundlePreset")
    .map { value -> value.trim().lowercase().replace("-", "_") }
    .orElse("")
    .get()

val requestedModelFamilyBundlePreset = if (requestedModelFamilyBundlePresetId.isBlank()) {
    null
} else {
    modelFamilyBundlePresetMap[requestedModelFamilyBundlePresetId]
        ?: throw GradleException(
            "Unknown gollek.modelFamilyBundlePreset '${requestedModelFamilyBundlePresetId}'. " +
                    "Use presets (${modelFamilyBundlePresetIds.sorted().joinToString(", ")})."
        )
}

val requestedModelFamilyProperty = providers.gradleProperty("gollek.modelFamilies")
val requestedModelFamilyTokens = requestedModelFamilyProperty
    .map { value -> parseModelFamilySelectorTokens(value, setOf("none")) }
    .orElse(requestedModelFamilyBundlePreset?.selectors ?: setOf("core"))
    .get()
val explicitModelFamilyTokens = requestedModelFamilyProperty
    .map { value -> parseModelFamilySelectorTokens(value, setOf("none")) }
    .orElse(emptySet())
    .get()
val presetModelFamilyTokens = requestedModelFamilyBundlePreset?.selectors ?: emptySet()
val defaultModelFamilyTokens = setOf("core")
val requestedModelFamilySelectorSource = requestedModelFamilyProperty
    .map { "explicit" }
    .orElse(if (requestedModelFamilyBundlePreset == null) "default" else "preset")
    .get()

val configuredRequiredModelFamilyTokens = providers.gradleProperty("gollek.requiredModelFamilies")
    .map { value -> parseModelFamilySelectorTokens(value, emptySet()) }
    .orElse(emptySet())
    .get()

val configuredForbiddenModelFamilyTokens = providers.gradleProperty("gollek.forbiddenModelFamilies")
    .map { value -> parseModelFamilySelectorTokens(value, emptySet()) }
    .orElse(emptySet())
    .get()

val configuredRequiredModelFamilyAliasTokens = providers.gradleProperty("gollek.requiredModelFamilyAliases")
    .map { value -> parseModelFamilySelectorTokens(value, emptySet()) }
    .orElse(emptySet())
    .get()

val configuredForbiddenModelFamilyAliasTokens = providers.gradleProperty("gollek.forbiddenModelFamilyAliases")
    .map { value -> parseModelFamilySelectorTokens(value, emptySet()) }
    .orElse(emptySet())
    .get()

val presetRequiredModelFamilyTokens = requestedModelFamilyBundlePreset?.requiredFamilies ?: emptySet()
val presetForbiddenModelFamilyTokens = requestedModelFamilyBundlePreset?.forbiddenFamilies ?: emptySet()
val presetRequiredModelFamilyAliasTokens = requestedModelFamilyBundlePreset?.requiredAliases ?: emptySet()
val presetForbiddenModelFamilyAliasTokens = requestedModelFamilyBundlePreset?.forbiddenAliases ?: emptySet()
val explicitRequiredModelFamilyTokens = configuredRequiredModelFamilyTokens
val explicitForbiddenModelFamilyTokens = configuredForbiddenModelFamilyTokens
val explicitRequiredModelFamilyAliasTokens = configuredRequiredModelFamilyAliasTokens
val explicitForbiddenModelFamilyAliasTokens = configuredForbiddenModelFamilyAliasTokens
val hasPresetModelFamilyPolicy =
    presetRequiredModelFamilyTokens.isNotEmpty() ||
            presetForbiddenModelFamilyTokens.isNotEmpty() ||
            presetRequiredModelFamilyAliasTokens.isNotEmpty() ||
            presetForbiddenModelFamilyAliasTokens.isNotEmpty()
val hasExplicitModelFamilyPolicy =
    explicitRequiredModelFamilyTokens.isNotEmpty() ||
            explicitForbiddenModelFamilyTokens.isNotEmpty() ||
            explicitRequiredModelFamilyAliasTokens.isNotEmpty() ||
            explicitForbiddenModelFamilyAliasTokens.isNotEmpty()
val modelFamilyPolicySource = when {
    hasPresetModelFamilyPolicy && hasExplicitModelFamilyPolicy -> "mixed"
    hasPresetModelFamilyPolicy -> "preset"
    hasExplicitModelFamilyPolicy -> "explicit"
    else -> "none"
}
val requiredModelFamilyTokens = presetRequiredModelFamilyTokens + explicitRequiredModelFamilyTokens
val forbiddenModelFamilyTokens = presetForbiddenModelFamilyTokens + explicitForbiddenModelFamilyTokens
val requiredModelFamilyAliasTokens = presetRequiredModelFamilyAliasTokens + explicitRequiredModelFamilyAliasTokens
val forbiddenModelFamilyAliasTokens = presetForbiddenModelFamilyAliasTokens + explicitForbiddenModelFamilyAliasTokens
val requiresDirectSafetensorRuntime =
    requestedModelFamilyBundlePreset?.id == "prod_llm" ||
            ("direct" in requestedModelFamilyTokens && "direct" in requiredModelFamilyAliasTokens)

val reservedModelFamilySelectors = setOf("all", "none")

fun selectedModelFamilyIdsForSelectors(selectors: Set<String>): Set<String> {
    val expandedSelectors = expandModelFamilyBundleAliases(selectors)
    return modelFamilyModules
        .filter { module ->
            "all" in selectors
                    || module.id in expandedSelectors
                    || module.profile in selectors
        }
        .map { it.id }
        .toSet()
}

val bundledModelFamilies = selectedModelFamilyIdsForSelectors(requestedModelFamilyTokens)

val validModelFamilySelectors = modelFamilyModules
    .flatMap { listOf(it.id, it.profile) }
    .toSet() + modelFamilyBundleAliasIds + reservedModelFamilySelectors
val validModelFamilyPolicySelectors = modelFamilyModules
    .flatMap { listOf(it.id, it.profile) }
    .toSet() + modelFamilyBundleAliasIds
val invalidModelFamilyBundlePresetSelectors = modelFamilyBundlePresets
    .flatMap { preset ->
        preset.selectors.filter { selector -> selector !in validModelFamilySelectors }
            .map { selector -> "${preset.id}:selector:$selector" }
    }
val invalidModelFamilyBundlePresetFamilyPolicies = modelFamilyBundlePresets
    .flatMap { preset ->
        (preset.requiredFamilies + preset.forbiddenFamilies)
            .filter { selector -> selector !in validModelFamilyPolicySelectors }
            .map { selector -> "${preset.id}:family-policy:$selector" }
    }
val invalidModelFamilyBundlePresetAliasPolicies = modelFamilyBundlePresets
    .flatMap { preset ->
        (preset.requiredAliases + preset.forbiddenAliases)
            .filter { aliasId -> aliasId !in modelFamilyBundleAliasIds }
            .map { aliasId -> "${preset.id}:alias-policy:$aliasId" }
    }
val invalidModelFamilyBundlePresetEntries = invalidModelFamilyBundlePresetSelectors +
        invalidModelFamilyBundlePresetFamilyPolicies +
        invalidModelFamilyBundlePresetAliasPolicies
if (invalidModelFamilyBundlePresetEntries.isNotEmpty()) {
    throw GradleException(
        "Model-family bundle preset references unknown selectors: " +
                invalidModelFamilyBundlePresetEntries.sorted().joinToString(", ")
    )
}
val unknownModelFamilySelectors = requestedModelFamilyTokens - validModelFamilySelectors
if (unknownModelFamilySelectors.isNotEmpty()) {
    throw GradleException(
        "Unknown gollek.modelFamilies selector(s): ${unknownModelFamilySelectors.sorted().joinToString(", ")}. " +
                "Use family IDs (${modelFamilyModules.map { it.id }.sorted().joinToString(", ")}), " +
                "profiles (${modelFamilyModules.map { it.profile }.distinct().sorted().joinToString(", ")}), " +
                "aliases (${modelFamilyBundleAliasIds.sorted().joinToString(", ")}), all, or none."
    )
}
if ("none" in requestedModelFamilyTokens && requestedModelFamilyTokens.size > 1) {
    throw GradleException(
        "gollek.modelFamilies selector 'none' cannot be combined with other selectors. " +
                "Use either -Pgollek.modelFamilies=none to detach all model-family plugins, " +
                "or choose family IDs/profiles/all to package a model farm bundle."
    )
}
val unknownRequiredModelFamilySelectors = requiredModelFamilyTokens - validModelFamilyPolicySelectors
if (unknownRequiredModelFamilySelectors.isNotEmpty()) {
    throw GradleException(
        "Unknown gollek.requiredModelFamilies selector(s): ${unknownRequiredModelFamilySelectors.sorted().joinToString(", ")}. " +
        "Use family IDs (${modelFamilyModules.map { it.id }.sorted().joinToString(", ")}) " +
                "profiles (${modelFamilyModules.map { it.profile }.distinct().sorted().joinToString(", ")}), " +
                "or aliases (${modelFamilyBundleAliasIds.sorted().joinToString(", ")})."
    )
}
val unknownForbiddenModelFamilySelectors = forbiddenModelFamilyTokens - validModelFamilyPolicySelectors
if (unknownForbiddenModelFamilySelectors.isNotEmpty()) {
    throw GradleException(
        "Unknown gollek.forbiddenModelFamilies selector(s): ${unknownForbiddenModelFamilySelectors.sorted().joinToString(", ")}. " +
        "Use family IDs (${modelFamilyModules.map { it.id }.sorted().joinToString(", ")}) " +
                "profiles (${modelFamilyModules.map { it.profile }.distinct().sorted().joinToString(", ")}), " +
                "or aliases (${modelFamilyBundleAliasIds.sorted().joinToString(", ")})."
    )
}
val unknownRequiredModelFamilyAliases = requiredModelFamilyAliasTokens - modelFamilyBundleAliasIds
if (unknownRequiredModelFamilyAliases.isNotEmpty()) {
    throw GradleException(
        "Unknown gollek.requiredModelFamilyAliases alias selector(s): " +
                unknownRequiredModelFamilyAliases.sorted().joinToString(", ") +
                ". Use aliases (${modelFamilyBundleAliasIds.sorted().joinToString(", ")})."
    )
}
val unknownForbiddenModelFamilyAliases = forbiddenModelFamilyAliasTokens - modelFamilyBundleAliasIds
if (unknownForbiddenModelFamilyAliases.isNotEmpty()) {
    throw GradleException(
        "Unknown gollek.forbiddenModelFamilyAliases alias selector(s): " +
                unknownForbiddenModelFamilyAliases.sorted().joinToString(", ") +
                ". Use aliases (${modelFamilyBundleAliasIds.sorted().joinToString(", ")})."
    )
}
val conflictingModelFamilyPolicyAliases = requiredModelFamilyAliasTokens.intersect(forbiddenModelFamilyAliasTokens)
if (conflictingModelFamilyPolicyAliases.isNotEmpty()) {
    throw GradleException(
        "Model-family bundle alias policy conflict: aliases cannot be both required and forbidden: " +
                conflictingModelFamilyPolicyAliases.sorted().joinToString(", ")
    )
}

fun expandModelFamilyPolicyTokens(tokens: Set<String>): Set<String> {
    if (tokens.isEmpty()) {
        return emptySet()
    }
    val expandedTokens = expandModelFamilyBundleAliases(tokens)
    return modelFamilyModules
        .filter { module -> module.id in expandedTokens || module.profile in tokens }
        .map { it.id }
        .toSet()
}

val requiredModelFamilies = expandModelFamilyPolicyTokens(requiredModelFamilyTokens)
val forbiddenModelFamilies = expandModelFamilyPolicyTokens(forbiddenModelFamilyTokens)
val conflictingModelFamilyPolicyFamilies = requiredModelFamilies.intersect(forbiddenModelFamilies)
if (conflictingModelFamilyPolicyFamilies.isNotEmpty()) {
    throw GradleException(
        "Model-family bundle policy conflict: families cannot be both required and forbidden: " +
                conflictingModelFamilyPolicyFamilies.sorted().joinToString(", ")
    )
}

val validModelFamilyFixtureSelectors = validModelFamilyPolicySelectors + reservedModelFamilySelectors
val requiredModelFamilyFixtureTokens = providers.gradleProperty("gollek.requiredModelFamilyFixtures")
    .map { value -> parseModelFamilySelectorTokens(value, setOf("core")) }
    .orElse(setOf("core"))
    .get()
val unknownRequiredModelFamilyFixtureSelectors =
    requiredModelFamilyFixtureTokens - validModelFamilyFixtureSelectors
if (unknownRequiredModelFamilyFixtureSelectors.isNotEmpty()) {
    throw GradleException(
        "Unknown gollek.requiredModelFamilyFixtures selector(s): " +
                unknownRequiredModelFamilyFixtureSelectors.sorted().joinToString(", ") +
                ". Use family IDs (${modelFamilyModules.map { it.id }.sorted().joinToString(", ")}), " +
                "profiles (${modelFamilyModules.map { it.profile }.distinct().sorted().joinToString(", ")}), " +
                "aliases (${modelFamilyBundleAliasIds.sorted().joinToString(", ")}), all, or none."
    )
}
if ("none" in requiredModelFamilyFixtureTokens && requiredModelFamilyFixtureTokens.size > 1) {
    throw GradleException(
        "gollek.requiredModelFamilyFixtures selector 'none' cannot be combined with other selectors."
    )
}
val requiredModelFamilyFixtures = if ("none" in requiredModelFamilyFixtureTokens) {
    emptySet()
} else {
    selectedModelFamilyIdsForSelectors(requiredModelFamilyFixtureTokens)
}

fun jsonStringValue(json: String, key: String): String? {
    return Regex(""""$key"\s*:\s*"([^"]+)"""")
        .find(json)
        ?.groupValues
        ?.get(1)
}

val scopedDirectSafetensorMetadataPattern =
    Regex(""""([^"]+_direct_safetensor)"\s*,\s*"([^"]*)"""")
val scopedDirectSafetensorKeyPattern = Regex("""[a-z][a-z0-9_]*_direct_safetensor""")

fun isScopedDirectSafetensorReason(value: String): Boolean {
    val normalized = value.trim().lowercase()
    return normalized.startsWith("pending")
            || normalized.startsWith("experimental")
            || normalized.startsWith("not")
            || normalized.endsWith("_pending")
            || normalized.contains("_pending_")
}

fun validateScopedDirectSafetensorMetadata(
    module: ModelFamilyModule,
    sourceDir: File,
    problems: MutableList<String>
) {
    if (!sourceDir.isDirectory) {
        return
    }

    sourceDir.walkTopDown()
        .filter { it.isFile && it.extension == "java" }
        .forEach { sourceFile ->
            scopedDirectSafetensorMetadataPattern.findAll(sourceFile.readText(Charsets.UTF_8))
                .forEach { match ->
                    val key = match.groupValues[1]
                    val value = match.groupValues[2]
                    val location = sourceFile.relativeTo(rootDir)

                    if (!scopedDirectSafetensorKeyPattern.matches(key)) {
                        problems += "${module.id}: $location has malformed scoped direct SafeTensor key '$key'"
                    }
                    if (!isScopedDirectSafetensorReason(value)) {
                        problems += "${module.id}: $location metadata '$key' value '$value' must start with " +
                                "pending/experimental/not or include _pending"
                    }
                }
        }
}

val modelFamilyBundleManifestSchemaVersion = "1"
val modelFamilyBundleLockSchemaVersion = "6"
val modelFamilyFixtureFingerprintLockSchemaVersion = "1"
val modelFamilyBundleLockPath = providers.gradleProperty("gollek.modelFamilyBundleLock")
    .orElse("model-family-bundle.lock.properties")
    .get()
val modelFamilyBundleLockFile = layout.projectDirectory.file(modelFamilyBundleLockPath)
val modelFamilyFixtureFingerprintLockPath = providers.gradleProperty("gollek.modelFamilyFixtureFingerprintLock")
    .orElse("model-family-fixture-fingerprints.lock.properties")
    .get()
val modelFamilyFixtureFingerprintLockFile = layout.projectDirectory.file(modelFamilyFixtureFingerprintLockPath)
val enforceModelFamilyBundleLock = providers.gradleProperty("gollek.enforceModelFamilyBundleLock")
    .map { value -> value.toBoolean() }
    .orElse(false)
    .get()
val enforceModelFamilyFixtureFingerprintLock = providers.gradleProperty("gollek.enforceModelFamilyFixtureFingerprintLock")
    .map { value -> value.toBoolean() }
    .orElse(false)
    .get()
val enforceModelFamilyBundlePresetConformance = providers.gradleProperty("gollek.enforceModelFamilyBundlePresetConformance")
    .map { value -> value.toBoolean() }
    .orElse(false)
    .get()
val requireCleanModelFamilyBundlePresetConformance = providers.gradleProperty("gollek.requireCleanModelFamilyBundlePresetConformance")
    .map { value -> value.toBoolean() }
    .orElse(false)
    .get()

fun sha256Hex(bytes: ByteArray): String {
    val digest = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
    return digest.joinToString("") { byte: Byte ->
        (byte.toInt() and 0xff).toString(16).padStart(2, '0')
    }
}

fun sha256Hex(value: String): String {
    return sha256Hex(value.toByteArray(Charsets.UTF_8))
}

fun modelFamilyBundleFingerprint(selected: List<ModelFamilyModule>): String {
    val bundlePayload = if (selected.isEmpty()) {
        "detached\n"
    } else {
        selected.sortedBy { it.id }
            .joinToString("\n") { "${it.id}:${it.profile}:${it.path}" } + "\n"
    }
    return "sha256:" + sha256Hex(
        "gollek-model-family-bundle-v$modelFamilyBundleManifestSchemaVersion\n$bundlePayload"
    )
}

fun csvProperty(properties: Properties, key: String): List<String> {
    return properties.getProperty(key, "")
        .split(",")
        .map { it.trim() }
        .filter { it.isNotBlank() }
}

fun selectedModelFamilyModules(): List<ModelFamilyModule> {
    return modelFamilyModules.filter { it.id in bundledModelFamilies }
}

fun modelFamilyProject(module: ModelFamilyModule) = findProject(module.path)

fun existingModelFamilyModules(): List<ModelFamilyModule> {
    return modelFamilyModules.filter { module -> modelFamilyProject(module) != null }
}

fun catalogModelFamilyProjectPaths(): Set<String> {
    return modelFamilyModules.map { it.path }.toSet()
}

fun includedModelFamilyProjectPaths(): Set<String> {
    return rootProject.subprojects
        .map { it.path }
        .filter { it.startsWith(":models:gollek-model-") }
        .toSet()
}

fun includedCatalogedModelFamilyProjectPaths(): Set<String> {
    return rootProject.subprojects
        .filter { project ->
            project.path.startsWith(":models:gollek-model-") &&
                    project.layout.projectDirectory.file(modelFamilyPluginDescriptorRelativePath).asFile.isFile
        }
        .map { it.path }
        .toSet()
}

fun modelFamilyModuleCatalogMissingProjects(): Set<String> {
    return catalogModelFamilyProjectPaths() - includedModelFamilyProjectPaths()
}

fun modelFamilyModuleCatalogUncatalogedProjects(): Set<String> {
    return includedCatalogedModelFamilyProjectPaths() - catalogModelFamilyProjectPaths()
}

fun modelFamilyModuleCatalogSupportOnlyProjects(): Set<String> {
    return includedModelFamilyProjectPaths() - includedCatalogedModelFamilyProjectPaths()
}

fun catalogedIncludedModelFamilyProjectPaths(): Set<String> {
    return catalogModelFamilyProjectPaths().intersect(includedModelFamilyProjectPaths())
}

fun modelFamilyProjectResourceFile(projectPath: String, relativePath: String): File {
    return rootProject.findProject(projectPath)
        ?.layout
        ?.projectDirectory
        ?.file(relativePath)
        ?.asFile
        ?: rootDir
            .resolve(projectPath.trimStart(':').replace(':', File.separatorChar))
            .resolve(relativePath)
}

fun modelFamilyPluginDescriptorPresent(projectPath: String): Boolean {
    return modelFamilyProjectResourceFile(projectPath, modelFamilyPluginDescriptorRelativePath).isFile
}

fun modelFamilyServiceDescriptorPresent(projectPath: String): Boolean {
    return modelFamilyProjectResourceFile(projectPath, modelFamilyServiceDescriptorRelativePath).isFile
}

fun gollekPluginServiceDescriptorPresent(projectPath: String): Boolean {
    return modelFamilyProjectResourceFile(projectPath, gollekPluginServiceDescriptorRelativePath).isFile
}

fun modelFamilyServiceDescriptorsComplete(projectPath: String): Boolean {
    return modelFamilyPluginDescriptorPresent(projectPath) &&
            modelFamilyServiceDescriptorPresent(projectPath) &&
            gollekPluginServiceDescriptorPresent(projectPath)
}

fun modelFamilyPluginDescriptorTokenizerKinds(projectPath: String): List<String> {
    val pluginJson = modelFamilyProjectResourceFile(projectPath, modelFamilyPluginDescriptorRelativePath)
    if (!pluginJson.isFile) {
        return emptyList()
    }
    val json = pluginJson.readText()
    return (jsonFieldStringValues(json, "tokenizerKind") +
            jsonFieldStringArrayValues(json, "tokenizerKinds"))
        .distinct()
        .sorted()
}

fun modelFamilyPluginDescriptorHasTokenizerMetadata(projectPath: String): Boolean {
    return modelFamilyPluginDescriptorTokenizerKinds(projectPath).isNotEmpty()
}

fun modelFamilyPluginDescriptorStringValue(projectPath: String, key: String): String {
    val pluginJson = modelFamilyProjectResourceFile(projectPath, modelFamilyPluginDescriptorRelativePath)
    if (!pluginJson.isFile) {
        return ""
    }
    return jsonStringValue(pluginJson.readText(), key)?.trim().orEmpty()
}

fun modelFamilyPluginDescriptorTokenizerMetadataPending(projectPath: String): Boolean {
    return modelFamilyPluginDescriptorStringValue(projectPath, "tokenizerMetadataStatus")
        .equals("pending", ignoreCase = true)
}

fun modelFamilyPluginDescriptorTokenizerMetadataPendingReason(projectPath: String): String {
    return modelFamilyPluginDescriptorStringValue(projectPath, "tokenizerMetadataPendingReason")
}

fun modelFamilyTokenizerMetadataPendingReason(familyId: String): String {
    return pendingModelFamilyTokenizerMetadataReasons[familyId] ?: ""
}

fun pendingModelFamilyTokenizerMetadataInput(): String {
    return pendingModelFamilyTokenizerMetadataReasons.entries
        .sortedBy { it.key }
        .joinToString("|") { "${it.key}:${it.value}" }
}

fun modelFamilyModuleCatalogServiceIncompleteProjects(): Set<String> {
    return catalogedIncludedModelFamilyProjectPaths()
        .filterNot { modelFamilyServiceDescriptorsComplete(it) }
        .toSet()
}

fun modelFamilyModuleCatalogTokenizerMetadataMissingFamilies(): Set<String> {
    return modelFamilyModules
        .filter { module -> modelFamilyProject(module) != null }
        .filterNot { module -> module.id in pendingModelFamilyTokenizerMetadata }
        .filterNot { module -> modelFamilyPluginDescriptorHasTokenizerMetadata(module.path) }
        .map { it.id }
        .toSet()
}

fun modelFamilyModuleCatalogTokenizerMetadataPendingFamilies(): Set<String> {
    return modelFamilyModules
        .filter { module -> modelFamilyProject(module) != null }
        .filter { module -> module.id in pendingModelFamilyTokenizerMetadata }
        .map { it.id }
        .toSet()
}

fun modelFamilyModuleCatalogTokenizerMetadataPendingReasons(): Map<String, String> {
    return modelFamilyModuleCatalogTokenizerMetadataPendingFamilies()
        .associateWith(::modelFamilyTokenizerMetadataPendingReason)
}

val directSafetensorModelFamilies: Set<String> by lazy {
    modelFamilyBundleAliasMap["direct"]?.familyIds ?: emptySet()
}

fun modelFamilyModuleIncluded(module: ModelFamilyModule): Boolean {
    return modelFamilyProject(module) != null
}

fun modelFamilyModuleMetadataOnly(module: ModelFamilyModule): Boolean {
    return module.profile == "metadata_only"
}

fun modelFamilyModuleProductionCandidate(module: ModelFamilyModule): Boolean {
    return modelFamilyModuleIncluded(module) &&
            !modelFamilyModuleMetadataOnly(module) &&
            module.profile != "experimental"
}

fun modelFamilyModuleTokenizerReady(module: ModelFamilyModule): Boolean {
    return modelFamilyModuleIncluded(module) &&
            module.id !in pendingModelFamilyTokenizerMetadata &&
            modelFamilyPluginDescriptorHasTokenizerMetadata(module.path)
}

fun modelFamilyModuleProductionReady(module: ModelFamilyModule): Boolean {
    return modelFamilyModuleProductionCandidate(module) &&
            modelFamilyServiceDescriptorsComplete(module.path) &&
            modelFamilyModuleTokenizerReady(module)
}

fun modelFamilyModuleDirectSafetensorReady(module: ModelFamilyModule): Boolean {
    return module.id in directSafetensorModelFamilies &&
            modelFamilyModuleProductionReady(module)
}

fun modelFamilyModuleReadiness(module: ModelFamilyModule): String {
    val included = modelFamilyModuleIncluded(module)
    val tokenizerMetadataPending = module.id in pendingModelFamilyTokenizerMetadata
    val tokenizerMetadataPresent = modelFamilyPluginDescriptorHasTokenizerMetadata(module.path)
    return when {
        !included -> "not_included"
        !modelFamilyServiceDescriptorsComplete(module.path) -> "service_incomplete"
        tokenizerMetadataPending -> "tokenizer_pending"
        !tokenizerMetadataPresent -> "tokenizer_missing"
        modelFamilyModuleMetadataOnly(module) -> "metadata_only"
        module.profile == "experimental" -> "experimental_ready"
        modelFamilyModuleDirectSafetensorReady(module) -> "direct_safetensor_ready"
        modelFamilyModuleProductionReady(module) -> "production_ready"
        else -> "descriptor_ready"
    }
}

fun modelFamilyModuleCatalogProductionReadinessPendingFamilies(): Set<String> {
    return modelFamilyModules
        .filter(::modelFamilyModuleProductionCandidate)
        .filterNot(::modelFamilyModuleProductionReady)
        .map { it.id }
        .toSet()
}

fun modelFamilyModuleCatalogDirectSafetensorPendingFamilies(): Set<String> {
    return modelFamilyModules
        .filter { module -> module.id in directSafetensorModelFamilies }
        .filterNot(::modelFamilyModuleDirectSafetensorReady)
        .map { it.id }
        .toSet()
}

fun modelFamilyFallbackProjectDir(module: ModelFamilyModule): File {
    return rootDir.resolve(module.path.trimStart(':').replace(':', File.separatorChar))
}

fun modelFamilyProjectDir(module: ModelFamilyModule): File {
    return modelFamilyProject(module)?.layout?.projectDirectory?.asFile
        ?: modelFamilyFallbackProjectDir(module)
}

fun modelFamilyBundleAliasCoverage(selectedFamilyIds: Set<String>): List<ModelFamilyBundleAliasCoverage> {
    return modelFamilyBundleAliases.sortedBy { it.id }
        .map { alias ->
            ModelFamilyBundleAliasCoverage(
                alias,
                alias.familyIds.intersect(selectedFamilyIds),
                alias.familyIds - selectedFamilyIds
            )
        }
}

fun joinedModelFamilyBundleAliasCoverage(
    selectedFamilyIds: Set<String>,
    predicate: (ModelFamilyBundleAliasCoverage) -> Boolean
): String {
    return modelFamilyBundleAliasCoverage(selectedFamilyIds)
        .filter(predicate)
        .joinToString(", ") { it.compactSummary() }
        .ifBlank { "none" }
}

fun compactFamilyList(familyIds: Iterable<String>, limit: Int = 12): String {
    val sortedFamilyIds = familyIds.toList().sorted()
    if (sortedFamilyIds.isEmpty()) {
        return "none"
    }
    if (sortedFamilyIds.size <= limit) {
        return sortedFamilyIds.joinToString(", ")
    }
    return sortedFamilyIds.take(limit).joinToString(", ") +
            ", ... (+${sortedFamilyIds.size - limit} more)"
}

fun requestedModelFamilyFamilies(): List<String> {
    return requestedModelFamilyTokens
        .filter { it in modelFamilyIds }
        .sorted()
}

fun requestedModelFamilyProfiles(): List<String> {
    val profileIds = modelFamilyModules.map { it.profile }.toSet()
    return requestedModelFamilyTokens
        .filter { it in profileIds }
        .sorted()
}

fun requestedModelFamilyAliases(): List<String> {
    return requestedModelFamilyTokens
        .filter { it in modelFamilyBundleAliasIds }
        .sorted()
}

fun requestedModelFamilyReservedSelectors(): List<String> {
    return requestedModelFamilyTokens
        .filter { it in reservedModelFamilySelectors }
        .sorted()
}

fun modelFamilyPolicyMissingRequiredFor(selectedFamilyIds: Set<String>, requiredFamilyIds: Set<String>): Set<String> {
    return requiredFamilyIds - selectedFamilyIds
}

fun modelFamilyPolicyMissingRequired(): Set<String> {
    return modelFamilyPolicyMissingRequiredFor(bundledModelFamilies, requiredModelFamilies)
}

fun modelFamilyPolicySelectedForbiddenFor(selectedFamilyIds: Set<String>, forbiddenFamilyIds: Set<String>): Set<String> {
    return forbiddenFamilyIds.intersect(selectedFamilyIds)
}

fun modelFamilyPolicySelectedForbidden(): Set<String> {
    return modelFamilyPolicySelectedForbiddenFor(bundledModelFamilies, forbiddenModelFamilies)
}

fun modelFamilyPolicyMissingRequiredAliasesFor(
    selectedFamilyIds: Set<String>,
    aliasTokens: Set<String>
): Map<String, Set<String>> {
    return aliasTokens.sorted()
        .mapNotNull { aliasId ->
            val missing = modelFamilyBundleAliasMap.getValue(aliasId).familyIds - selectedFamilyIds
            if (missing.isEmpty()) null else aliasId to missing
        }
        .toMap()
}

fun modelFamilyPolicyMissingRequiredAliases(): Map<String, Set<String>> {
    return modelFamilyPolicyMissingRequiredAliasesFor(bundledModelFamilies, requiredModelFamilyAliasTokens)
}

fun modelFamilyPolicySelectedForbiddenAliasesFor(
    selectedFamilyIds: Set<String>,
    aliasTokens: Set<String>
): Map<String, Set<String>> {
    return aliasTokens.sorted()
        .mapNotNull { aliasId ->
            val selected = modelFamilyBundleAliasMap.getValue(aliasId).familyIds.intersect(selectedFamilyIds)
            if (selected.isEmpty()) null else aliasId to selected
        }
        .toMap()
}

fun modelFamilyPolicySelectedForbiddenAliases(): Map<String, Set<String>> {
    return modelFamilyPolicySelectedForbiddenAliasesFor(bundledModelFamilies, forbiddenModelFamilyAliasTokens)
}

fun modelFamilyPolicyViolationCount(
    missingRequired: Set<String>,
    selectedForbidden: Set<String>,
    missingRequiredAliases: Map<String, Set<String>>,
    selectedForbiddenAliases: Map<String, Set<String>>
): Int {
    return missingRequired.size +
            selectedForbidden.size +
            missingRequiredAliases.size +
            selectedForbiddenAliases.size
}

fun modelFamilyBundlePresetValidation(preset: ModelFamilyBundlePreset): ModelFamilyBundlePresetValidation {
    val selectedFamilyIds = selectedModelFamilyIdsForSelectors(preset.selectors)
    val requiredFamilyIds = expandModelFamilyPolicyTokens(preset.requiredFamilies)
    val forbiddenFamilyIds = expandModelFamilyPolicyTokens(preset.forbiddenFamilies)
    val pendingTokenizerFamilyIds = selectedFamilyIds.intersect(pendingModelFamilyTokenizerMetadata)
    val configurationProblems = mutableListOf<String>()

    if ("none" in preset.selectors && preset.selectors.size > 1) {
        configurationProblems += "selector 'none' cannot be combined with other selectors"
    }

    val conflictingAliasPolicies = preset.requiredAliases.intersect(preset.forbiddenAliases)
    if (conflictingAliasPolicies.isNotEmpty()) {
        configurationProblems += "aliases cannot be both required and forbidden: ${
            conflictingAliasPolicies.sorted().joinToString(", ")
        }"
    }

    val conflictingFamilyPolicies = requiredFamilyIds.intersect(forbiddenFamilyIds)
    if (conflictingFamilyPolicies.isNotEmpty()) {
        configurationProblems += "families cannot be both required and forbidden: ${
            conflictingFamilyPolicies.sorted().joinToString(", ")
        }"
    }

    return ModelFamilyBundlePresetValidation(
        preset = preset,
        selectedFamilyIds = selectedFamilyIds,
        productionTokenizerMetadataRequired = preset.id.startsWith("prod_"),
        pendingTokenizerFamilyIds = pendingTokenizerFamilyIds,
        pendingTokenizerReasons = pendingTokenizerFamilyIds.associateWith(::modelFamilyTokenizerMetadataPendingReason),
        configurationProblems = configurationProblems,
        missingRequiredFamilyIds = modelFamilyPolicyMissingRequiredFor(selectedFamilyIds, requiredFamilyIds),
        selectedForbiddenFamilyIds = modelFamilyPolicySelectedForbiddenFor(selectedFamilyIds, forbiddenFamilyIds),
        missingRequiredAliases = modelFamilyPolicyMissingRequiredAliasesFor(selectedFamilyIds, preset.requiredAliases),
        selectedForbiddenAliases = modelFamilyPolicySelectedForbiddenAliasesFor(selectedFamilyIds, preset.forbiddenAliases)
    )
}

fun modelFamilyBundlePresetValidations(): List<ModelFamilyBundlePresetValidation> {
    return modelFamilyBundlePresets.sortedBy { it.id }
        .map(::modelFamilyBundlePresetValidation)
}

fun normalizedSet(values: Iterable<String>): Set<String> {
    return values.map { it.trim().lowercase().replace("-", "_") }
        .filter { it.isNotBlank() }
        .toSet()
}

fun modelFamilyBundlePresetConformanceValidation(): ModelFamilyBundlePresetConformanceValidation {
    val presetId = requestedModelFamilyBundlePreset?.id ?: ""
    val preset = requestedModelFamilyBundlePreset
    if (preset == null) {
        return ModelFamilyBundlePresetConformanceValidation(
            presetId = "",
            presetMetadataPresent = false,
            selectorsMatch = false,
            policyInputsMatch = false,
            explicitSelectorOverride = requestedModelFamilySelectorSource == "explicit",
            explicitPolicyOverride = modelFamilyPolicySource == "explicit" || modelFamilyPolicySource == "mixed",
            selectorAdditions = emptySet(),
            selectorOmissions = emptySet(),
            requiredFamilyAdditions = emptySet(),
            requiredFamilyOmissions = emptySet(),
            forbiddenFamilyAdditions = emptySet(),
            forbiddenFamilyOmissions = emptySet(),
            requiredAliasAdditions = emptySet(),
            requiredAliasOmissions = emptySet(),
            forbiddenAliasAdditions = emptySet(),
            forbiddenAliasOmissions = emptySet()
        )
    }

    val selectorAdditions = normalizedSet(requestedModelFamilyTokens) - normalizedSet(preset.selectors)
    val selectorOmissions = normalizedSet(preset.selectors) - normalizedSet(requestedModelFamilyTokens)
    val requiredFamilyAdditions = normalizedSet(requiredModelFamilyTokens) - normalizedSet(preset.requiredFamilies)
    val requiredFamilyOmissions = normalizedSet(preset.requiredFamilies) - normalizedSet(requiredModelFamilyTokens)
    val forbiddenFamilyAdditions = normalizedSet(forbiddenModelFamilyTokens) - normalizedSet(preset.forbiddenFamilies)
    val forbiddenFamilyOmissions = normalizedSet(preset.forbiddenFamilies) - normalizedSet(forbiddenModelFamilyTokens)
    val requiredAliasAdditions = normalizedSet(requiredModelFamilyAliasTokens) - normalizedSet(preset.requiredAliases)
    val requiredAliasOmissions = normalizedSet(preset.requiredAliases) - normalizedSet(requiredModelFamilyAliasTokens)
    val forbiddenAliasAdditions = normalizedSet(forbiddenModelFamilyAliasTokens) - normalizedSet(preset.forbiddenAliases)
    val forbiddenAliasOmissions = normalizedSet(preset.forbiddenAliases) - normalizedSet(forbiddenModelFamilyAliasTokens)

    return ModelFamilyBundlePresetConformanceValidation(
        presetId = presetId,
        presetMetadataPresent = true,
        selectorsMatch = selectorAdditions.isEmpty() && selectorOmissions.isEmpty(),
        policyInputsMatch = requiredFamilyAdditions.isEmpty() &&
                requiredFamilyOmissions.isEmpty() &&
                forbiddenFamilyAdditions.isEmpty() &&
                forbiddenFamilyOmissions.isEmpty() &&
                requiredAliasAdditions.isEmpty() &&
                requiredAliasOmissions.isEmpty() &&
                forbiddenAliasAdditions.isEmpty() &&
                forbiddenAliasOmissions.isEmpty(),
        explicitSelectorOverride = requestedModelFamilySelectorSource == "explicit",
        explicitPolicyOverride = modelFamilyPolicySource == "explicit" || modelFamilyPolicySource == "mixed",
        selectorAdditions = selectorAdditions,
        selectorOmissions = selectorOmissions,
        requiredFamilyAdditions = requiredFamilyAdditions,
        requiredFamilyOmissions = requiredFamilyOmissions,
        forbiddenFamilyAdditions = forbiddenFamilyAdditions,
        forbiddenFamilyOmissions = forbiddenFamilyOmissions,
        requiredAliasAdditions = requiredAliasAdditions,
        requiredAliasOmissions = requiredAliasOmissions,
        forbiddenAliasAdditions = forbiddenAliasAdditions,
        forbiddenAliasOmissions = forbiddenAliasOmissions
    )
}

fun jsonString(value: String): String {
    val escaped = buildString {
        for (char in value) {
            when (char) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> {
                    if (char.code < 0x20) {
                        append("\\u")
                        append(char.code.toString(16).padStart(4, '0'))
                    } else {
                        append(char)
                    }
                }
            }
        }
    }
    return "\"$escaped\""
}

fun jsonStringArray(values: Iterable<String>): String {
    return values.joinToString(prefix = "[", postfix = "]") { jsonString(it) }
}

fun jsonStringMap(values: Map<String, String>): String {
    return values.entries.sortedBy { it.key }
        .joinToString(prefix = "{", postfix = "}") { (key, value) ->
            "${jsonString(key)}: ${jsonString(value)}"
        }
}

fun jsonIntMap(values: Map<String, Int>): String {
    return values.entries.sortedBy { it.key }
        .joinToString(prefix = "{", postfix = "}") { (key, value) ->
            "${jsonString(key)}: $value"
        }
}

fun jsonAliasPolicyViolations(values: Map<String, Set<String>>, familyField: String): String {
    return values.entries.sortedBy { it.key }
        .joinToString(prefix = "[", postfix = "]") { (aliasId, families) ->
            """{"alias": ${jsonString(aliasId)}, "$familyField": ${jsonStringArray(families.sorted())}}"""
        }
}

val modelFamilyFixtureTokenizerMarkers = listOf(
    "tokenizer.json",
    "tokenizer/tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "vocab.txt",
    "tokenizer/vocab.txt",
    "tokenizer.model",
    "tokenizer/tokenizer.model",
    "spiece.model",
    "tokenizer/spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer/sentencepiece.bpe.model"
)

fun jsonFieldStringValues(json: String, key: String): List<String> {
    return Regex(""""$key"\s*:\s*"([^"]+)"""")
        .findAll(json)
        .map { it.groupValues[1].trim() }
        .filter { it.isNotBlank() }
        .distinct()
        .sorted()
        .toList()
}

fun jsonFieldStringArrayValues(json: String, key: String): List<String> {
    val quotedValue = Regex(""""([^"]+)"""")
    return Regex(""""$key"\s*:\s*\[([^\]]*)]""")
        .findAll(json)
        .flatMap { match -> quotedValue.findAll(match.groupValues[1]).map { it.groupValues[1].trim() } }
        .filter { it.isNotBlank() }
        .distinct()
        .sorted()
        .toList()
}

fun modelFamilyFixtureDir(module: ModelFamilyModule): File {
    return modelFamilyProjectDir(module).resolve("src/test/resources/model-family-fixtures/${module.id}")
}

fun relativeRootPath(file: File): String {
    return file.relativeTo(rootDir).path.replace(File.separatorChar, '/')
}

fun modelFamilyFixtureTokenizerFiles(fixtureDir: File): List<String> {
    if (!fixtureDir.isDirectory) {
        return emptyList()
    }
    return modelFamilyFixtureTokenizerMarkers
        .filter { marker -> fixtureDir.resolve(marker).isFile }
        .sorted()
}

fun modelFamilyFixtureFiles(fixtureDir: File): List<File> {
    if (!fixtureDir.isDirectory) {
        return emptyList()
    }
    return fixtureDir.walkTopDown()
        .filter { it.isFile }
        .sortedBy { it.relativeTo(fixtureDir).path.replace(File.separatorChar, '/') }
        .toList()
}

fun modelFamilyFixtureContentFingerprint(fixtureDir: File): String {
    if (!fixtureDir.isDirectory) {
        return "absent"
    }
    val payload = buildString {
        appendLine("gollek-model-family-fixture-content-v1")
        modelFamilyFixtureFiles(fixtureDir).forEach { file ->
            val relativePath = file.relativeTo(fixtureDir).path.replace(File.separatorChar, '/')
            append(relativePath)
            append("=")
            append(sha256Hex(file.readBytes()))
            appendLine()
        }
    }
    return "sha256:" + sha256Hex(payload)
}

fun modelFamilyFixtureInventory(module: ModelFamilyModule): ModelFamilyFixtureInventory {
    val fixtureDir = modelFamilyFixtureDir(module)
    val present = fixtureDir.isDirectory
    val config = fixtureDir.resolve("config.json")
    val configPresent = config.isFile
    val configJson = if (configPresent) config.readText(Charsets.UTF_8) else ""
    val modelTypes = jsonFieldStringValues(configJson, "model_type")
    val architectures = jsonFieldStringArrayValues(configJson, "architectures") +
            jsonFieldStringValues(configJson, "architectures").filter { it !in jsonFieldStringArrayValues(configJson, "architectures") }
    val tokenizerFiles = modelFamilyFixtureTokenizerFiles(fixtureDir)
    val required = module.id in requiredModelFamilyFixtures
    val problems = mutableListOf<String>()

    if (!present) {
        if (required) {
            problems += "missing fixture directory"
        }
    } else {
        if (!configPresent) {
            problems += "missing config.json"
        } else {
            if (modelTypes.isEmpty()) {
                problems += "config.json has no model_type"
            }
            if (architectures.isEmpty()) {
                problems += "config.json has no architectures"
            }
        }
        if (tokenizerFiles.isEmpty()) {
            problems += "missing tokenizer marker file"
        }
    }

    return ModelFamilyFixtureInventory(
        module = module,
        selected = module.id in bundledModelFamilies,
        required = required,
        fixturePath = relativeRootPath(fixtureDir),
        present = present,
        configPresent = configPresent,
        modelTypes = modelTypes,
        architectures = architectures.distinct().sorted(),
        tokenizerFiles = tokenizerFiles,
        contentFingerprint = modelFamilyFixtureContentFingerprint(fixtureDir),
        problems = problems
    )
}

fun modelFamilyFixtureInventories(): List<ModelFamilyFixtureInventory> {
    return modelFamilyModules.sortedBy { it.id }.map(::modelFamilyFixtureInventory)
}

fun modelFamilyFixtureInventoryFingerprint(
    scope: String,
    inventories: List<ModelFamilyFixtureInventory>
): String {
    val payload = buildString {
        appendLine("gollek-model-family-fixture-inventory-v1")
        appendLine("scope=$scope")
        inventories.sortedBy { it.module.id }.forEach { inventory ->
            append(inventory.module.id)
            append("|profile=")
            append(inventory.module.profile)
            append("|path=")
            append(inventory.module.path)
            append("|selected=")
            append(inventory.selected)
            append("|required=")
            append(inventory.required)
            append("|present=")
            append(inventory.present)
            append("|configPresent=")
            append(inventory.configPresent)
            append("|modelTypes=")
            append(inventory.modelTypes.joinToString(","))
            append("|architectures=")
            append(inventory.architectures.joinToString(","))
            append("|tokenizerFiles=")
            append(inventory.tokenizerFiles.joinToString(","))
            append("|content=")
            append(inventory.contentFingerprint)
            append("|problems=")
            append(inventory.problems.joinToString(","))
            appendLine()
        }
    }
    return "sha256:" + sha256Hex(payload)
}

fun modelFamilyFixtureContentFingerprintMap(inventories: List<ModelFamilyFixtureInventory>): Map<String, String> {
    return inventories
        .sortedBy { it.module.id }
        .associate { it.module.id to it.contentFingerprint }
}

fun modelFamilyFixtureFingerprintMapValue(fingerprints: Map<String, String>): String {
    return fingerprints.entries
        .sortedBy { it.key }
        .joinToString(",") { (familyId, fingerprint) -> "$familyId=$fingerprint" }
}

fun jsonModelFamilyFixtureFingerprintEntries(fingerprints: Map<String, String>): String {
    return fingerprints.entries
        .sortedBy { it.key }
        .joinToString(prefix = "[", postfix = "]") { (familyId, fingerprint) ->
            """{"id": ${jsonString(familyId)}, "contentFingerprint": ${jsonString(fingerprint)}}"""
        }
}

fun modelFamilyFixtureLockSnapshot(
    inventories: List<ModelFamilyFixtureInventory> = modelFamilyFixtureInventories()
): ModelFamilyFixtureLockSnapshot {
    val requiredInventories = inventories.filter { it.required }
    val problemInventories = inventories.filter { it.problems.isNotEmpty() }
    val missingRequired = requiredInventories.filter { !it.passed }.map { it.module.id }.sorted()
    return ModelFamilyFixtureLockSnapshot(
        requiredSelectors = requiredModelFamilyFixtureTokens.sorted(),
        requiredFamilies = requiredModelFamilyFixtures.sorted(),
        passed = missingRequired.isEmpty(),
        availableFamilyCount = inventories.size,
        fixtureFamilyCount = inventories.count { it.present },
        requiredFamilyCount = requiredInventories.size,
        requiredPassedCount = requiredInventories.count { it.passed },
        missingRequiredCount = missingRequired.size,
        problemFamilyCount = problemInventories.size,
        missingRequiredFamilies = missingRequired,
        problemFamilies = problemInventories.map { it.module.id }.sorted(),
        requiredFingerprint = modelFamilyFixtureInventoryFingerprint("required", requiredInventories),
        inventoryFingerprint = modelFamilyFixtureInventoryFingerprint("all", inventories),
        requiredContentFingerprints = modelFamilyFixtureContentFingerprintMap(requiredInventories),
        presentContentFingerprints = modelFamilyFixtureContentFingerprintMap(inventories.filter { it.present })
    )
}

fun jsonModelFamilyFixtureInventory(inventory: ModelFamilyFixtureInventory): String {
    return buildString {
        append("{")
        append("\"id\": ${jsonString(inventory.module.id)}, ")
        append("\"profile\": ${jsonString(inventory.module.profile)}, ")
        append("\"path\": ${jsonString(inventory.module.path)}, ")
        append("\"selected\": ${inventory.selected}, ")
        append("\"required\": ${inventory.required}, ")
        append("\"fixturePath\": ${jsonString(inventory.fixturePath)}, ")
        append("\"present\": ${inventory.present}, ")
        append("\"configPresent\": ${inventory.configPresent}, ")
        append("\"modelTypes\": ${jsonStringArray(inventory.modelTypes)}, ")
        append("\"architectures\": ${jsonStringArray(inventory.architectures)}, ")
        append("\"tokenizerFiles\": ${jsonStringArray(inventory.tokenizerFiles)}, ")
        append("\"contentFingerprint\": ${jsonString(inventory.contentFingerprint)}, ")
        append("\"passed\": ${inventory.passed}, ")
        append("\"problems\": ${jsonStringArray(inventory.problems)}")
        append("}")
    }
}

fun jsonModelFamilyFixtureCoverage(inventories: List<ModelFamilyFixtureInventory>): String {
    val snapshot = modelFamilyFixtureLockSnapshot(inventories)
    return buildString {
        append("{")
        append("\"availableFamilyCount\": ${snapshot.availableFamilyCount}, ")
        append("\"fixtureFamilyCount\": ${snapshot.fixtureFamilyCount}, ")
        append("\"requiredFamilyCount\": ${snapshot.requiredFamilyCount}, ")
        append("\"requiredPassedCount\": ${snapshot.requiredPassedCount}, ")
        append("\"missingRequiredCount\": ${snapshot.missingRequiredCount}, ")
        append("\"problemFamilyCount\": ${snapshot.problemFamilyCount}")
        append("}")
    }
}

fun jsonModelFamilyFixtureStatus(inventories: List<ModelFamilyFixtureInventory>): String {
    val snapshot = modelFamilyFixtureLockSnapshot(inventories)
    return buildString {
        append("{")
        append("\"requiredSelectors\": ${jsonStringArray(snapshot.requiredSelectors)}, ")
        append("\"requiredFamilies\": ${jsonStringArray(snapshot.requiredFamilies)}, ")
        append("\"passed\": ${snapshot.passed}, ")
        append("\"requiredFingerprint\": ${jsonString(snapshot.requiredFingerprint)}, ")
        append("\"inventoryFingerprint\": ${jsonString(snapshot.inventoryFingerprint)}, ")
        append("\"requiredContentFingerprints\": ${jsonModelFamilyFixtureFingerprintEntries(snapshot.requiredContentFingerprints)}, ")
        append("\"presentContentFingerprints\": ${jsonModelFamilyFixtureFingerprintEntries(snapshot.presentContentFingerprints)}, ")
        append("\"coverage\": ${jsonModelFamilyFixtureCoverage(inventories)}, ")
        append("\"missingRequiredFamilies\": ${jsonStringArray(snapshot.missingRequiredFamilies)}, ")
        append("\"problemFamilies\": ${jsonStringArray(snapshot.problemFamilies)}")
        append("}")
    }
}

fun jsonModelFamilyFixtureSelection(inventory: ModelFamilyFixtureInventory): String {
    return buildString {
        append("{")
        append("\"required\": ${inventory.required}, ")
        append("\"present\": ${inventory.present}, ")
        append("\"configPresent\": ${inventory.configPresent}, ")
        append("\"passed\": ${inventory.passed}, ")
        append("\"tokenizerFiles\": ${jsonStringArray(inventory.tokenizerFiles)}, ")
        append("\"contentFingerprint\": ${jsonString(inventory.contentFingerprint)}, ")
        append("\"problems\": ${jsonStringArray(inventory.problems)}")
        append("}")
    }
}

fun modelFamilyFixtureReportJson(): String {
    val inventories = modelFamilyFixtureInventories()
    val snapshot = modelFamilyFixtureLockSnapshot(inventories)
    return buildString {
        appendLine("{")
        appendLine("  \"schemaVersion\": 1,")
        appendLine("  \"requiredSelectors\": ${jsonStringArray(snapshot.requiredSelectors)},")
        appendLine("  \"requiredFamilies\": ${jsonStringArray(snapshot.requiredFamilies)},")
        appendLine("  \"requiredFixtureFingerprint\": ${jsonString(snapshot.requiredFingerprint)},")
        appendLine("  \"fixtureInventoryFingerprint\": ${jsonString(snapshot.inventoryFingerprint)},")
        appendLine("  \"requiredContentFingerprints\": ${jsonModelFamilyFixtureFingerprintEntries(snapshot.requiredContentFingerprints)},")
        appendLine("  \"presentContentFingerprints\": ${jsonModelFamilyFixtureFingerprintEntries(snapshot.presentContentFingerprints)},")
        appendLine("  \"fixtureCoverage\": ${jsonModelFamilyFixtureCoverage(inventories)},")
        appendLine("  \"missingRequiredFamilies\": ${jsonStringArray(snapshot.missingRequiredFamilies)},")
        appendLine("  \"problemFamilies\": ${jsonStringArray(snapshot.problemFamilies)},")
        appendLine("  \"fixtures\": [")
        inventories.forEachIndexed { index, inventory ->
            append("    ${jsonModelFamilyFixtureInventory(inventory)}")
            appendLine(if (index == inventories.lastIndex) "" else ",")
        }
        appendLine("  ]")
        appendLine("}")
    }
}

fun modelFamilyFixtureFingerprintReportJson(): String {
    val inventories = modelFamilyFixtureInventories()
    val snapshot = modelFamilyFixtureLockSnapshot(inventories)
    return buildString {
        appendLine("{")
        appendLine("  \"schemaVersion\": 1,")
        appendLine("  \"requiredSelectors\": ${jsonStringArray(snapshot.requiredSelectors)},")
        appendLine("  \"requiredFamilies\": ${jsonStringArray(snapshot.requiredFamilies)},")
        appendLine("  \"passed\": ${snapshot.passed},")
        appendLine("  \"requiredFixtureFingerprint\": ${jsonString(snapshot.requiredFingerprint)},")
        appendLine("  \"fixtureInventoryFingerprint\": ${jsonString(snapshot.inventoryFingerprint)},")
        appendLine("  \"fixtureCoverage\": ${jsonModelFamilyFixtureCoverage(inventories)},")
        appendLine("  \"missingRequiredFamilies\": ${jsonStringArray(snapshot.missingRequiredFamilies)},")
        appendLine("  \"problemFamilies\": ${jsonStringArray(snapshot.problemFamilies)},")
        appendLine("  \"requiredContentFingerprints\": ${jsonModelFamilyFixtureFingerprintEntries(snapshot.requiredContentFingerprints)},")
        appendLine("  \"presentContentFingerprints\": ${jsonModelFamilyFixtureFingerprintEntries(snapshot.presentContentFingerprints)}")
        appendLine("}")
    }
}

fun jsonModelFamilyFixtureFingerprintStatus(): String {
    val inventories = modelFamilyFixtureInventories()
    val snapshot = modelFamilyFixtureLockSnapshot(inventories)
    return buildString {
        append("{")
        append("\"schemaVersion\": 1, ")
        append("\"requiredSelectors\": ${jsonStringArray(snapshot.requiredSelectors)}, ")
        append("\"requiredFamilies\": ${jsonStringArray(snapshot.requiredFamilies)}, ")
        append("\"passed\": ${snapshot.passed}, ")
        append("\"requiredFixtureFingerprint\": ${jsonString(snapshot.requiredFingerprint)}, ")
        append("\"fixtureInventoryFingerprint\": ${jsonString(snapshot.inventoryFingerprint)}, ")
        append("\"fixtureCoverage\": ${jsonModelFamilyFixtureCoverage(inventories)}, ")
        append("\"missingRequiredFamilies\": ${jsonStringArray(snapshot.missingRequiredFamilies)}, ")
        append("\"problemFamilies\": ${jsonStringArray(snapshot.problemFamilies)}, ")
        append("\"requiredContentFingerprints\": ${jsonModelFamilyFixtureFingerprintEntries(snapshot.requiredContentFingerprints)}, ")
        append("\"presentContentFingerprints\": ${jsonModelFamilyFixtureFingerprintEntries(snapshot.presentContentFingerprints)}")
        append("}")
    }
}

fun modelFamilyFixtureFingerprintLockEntries(): Map<String, String> {
    val snapshot = modelFamilyFixtureLockSnapshot()
    return linkedMapOf(
        "lockSchemaVersion" to modelFamilyFixtureFingerprintLockSchemaVersion,
        "fixtureRequiredSelectors" to snapshot.requiredSelectors.joinToString(","),
        "fixtureRequiredFamilies" to snapshot.requiredFamilies.joinToString(","),
        "fixturePassed" to snapshot.passed.toString(),
        "fixtureRequiredFingerprint" to snapshot.requiredFingerprint,
        "fixtureInventoryFingerprint" to snapshot.inventoryFingerprint,
        "fixtureRequiredContentFingerprints" to modelFamilyFixtureFingerprintMapValue(
            snapshot.requiredContentFingerprints
        ),
        "fixturePresentContentFingerprints" to modelFamilyFixtureFingerprintMapValue(
            snapshot.presentContentFingerprints
        ),
        "fixtureAvailableFamilyCount" to snapshot.availableFamilyCount.toString(),
        "fixtureFamilyCount" to snapshot.fixtureFamilyCount.toString(),
        "fixtureRequiredFamilyCount" to snapshot.requiredFamilyCount.toString(),
        "fixtureRequiredPassedCount" to snapshot.requiredPassedCount.toString(),
        "fixtureMissingRequiredCount" to snapshot.missingRequiredCount.toString(),
        "fixtureProblemFamilyCount" to snapshot.problemFamilyCount.toString(),
        "fixtureMissingRequiredFamilies" to snapshot.missingRequiredFamilies.joinToString(","),
        "fixtureProblemFamilies" to snapshot.problemFamilies.joinToString(",")
    )
}

fun modelFamilyFixtureFingerprintLockText(): String {
    return buildString {
        appendLine("# Generated by :ui:gollek-cli:writeModelFamilyFixtureFingerprintLock")
        appendLine("# Regenerate only after intentionally changing model-family fixture content.")
        modelFamilyFixtureFingerprintLockEntries().forEach { (key, value) ->
            appendLine("$key=$value")
        }
    }
}

fun modelFamilyFixtureFingerprintLockInput(): String {
    return modelFamilyFixtureFingerprintLockEntries()
        .entries
        .joinToString("\n") { (key, value) -> "$key=$value" }
}

fun modelFamilyTokenizerPendingReasonsValue(reasons: Map<String, String>): String {
    return reasons.entries
        .sortedBy { it.key }
        .joinToString(",") { (familyId, reason) -> "$familyId=$reason" }
}

fun jsonModelFamilyBundlePresets(): String {
    return modelFamilyBundlePresets.sortedBy { it.id }
        .joinToString(prefix = "[", postfix = "]") { preset ->
            val validation = modelFamilyBundlePresetValidation(preset)
            buildString {
                append("{")
                append("\"id\": ${jsonString(preset.id)}, ")
                append("\"description\": ${jsonString(preset.description)}, ")
                append("\"selectors\": ${jsonStringArray(preset.selectors.sorted())}, ")
                append("\"requiredFamilies\": ${jsonStringArray(preset.requiredFamilies.sorted())}, ")
                append("\"forbiddenFamilies\": ${jsonStringArray(preset.forbiddenFamilies.sorted())}, ")
                append("\"requiredAliases\": ${jsonStringArray(preset.requiredAliases.sorted())}, ")
                append("\"forbiddenAliases\": ${jsonStringArray(preset.forbiddenAliases.sorted())}, ")
                append("\"selectedFamilies\": ${jsonStringArray(validation.selectedFamilyIds.sorted())}, ")
                append("\"selectedCount\": ${validation.selectedFamilyIds.size}, ")
                append(
                    "\"productionSafety\": {\"tokenizerMetadataRequired\": ${
                        validation.productionTokenizerMetadataRequired
                    }, \"tokenizerMetadataReady\": ${validation.productionTokenizerMetadataReady}, " +
                            "\"passed\": ${validation.productionSafetyPassed}, " +
                            "\"violationCount\": ${validation.productionSafetyViolationCount}, " +
                            "\"pendingTokenizerFamilyCount\": ${validation.pendingTokenizerFamilyIds.size}, " +
                            "\"pendingTokenizerFamilies\": ${
                                jsonStringArray(validation.pendingTokenizerFamilyIds.sorted())
                            }, \"pendingTokenizerReasons\": ${jsonStringMap(validation.pendingTokenizerReasons)}}, "
                )
                append(
                    "\"policyStatus\": {\"passed\": ${validation.passed}, " +
                            "\"violationCount\": ${validation.violationCount}}, "
                )
                append("\"policyViolations\": {")
                append("\"configuration\": ${jsonStringArray(validation.configurationProblems)}, ")
                append("\"missingRequired\": ${jsonStringArray(validation.missingRequiredFamilyIds.sorted())}, ")
                append("\"selectedForbidden\": ${jsonStringArray(validation.selectedForbiddenFamilyIds.sorted())}, ")
                append(
                    "\"missingRequiredAliases\": ${
                        jsonAliasPolicyViolations(validation.missingRequiredAliases, "missingFamilies")
                    }, "
                )
                append(
                    "\"selectedForbiddenAliases\": ${
                        jsonAliasPolicyViolations(validation.selectedForbiddenAliases, "selectedFamilies")
                    }"
                )
                append("}")
                append("}")
            }
        }
}

fun jsonModelFamilyBundlePresetConformance(): String {
    val conformance = modelFamilyBundlePresetConformanceValidation()
    return buildString {
        append("{")
        append("\"presetId\": ${conformance.presetId.takeIf { it.isNotBlank() }?.let(::jsonString) ?: "null"}, ")
        append("\"presetMetadataPresent\": ${conformance.presetMetadataPresent}, ")
        append("\"status\": ${jsonString(conformance.statusLabel)}, ")
        append("\"summary\": ${jsonString(conformance.compactSummary)}, ")
        append("\"matchesPreset\": ${conformance.matchesPreset}, ")
        append("\"cleanPresetBuild\": ${conformance.cleanPresetBuild}, ")
        append("\"selectorsMatch\": ${conformance.selectorsMatch}, ")
        append("\"policyInputsMatch\": ${conformance.policyInputsMatch}, ")
        append("\"explicitSelectorOverride\": ${conformance.explicitSelectorOverride}, ")
        append("\"explicitPolicyOverride\": ${conformance.explicitPolicyOverride}, ")
        append("\"selectorAdditions\": ${jsonStringArray(conformance.selectorAdditions.sorted())}, ")
        append("\"selectorOmissions\": ${jsonStringArray(conformance.selectorOmissions.sorted())}, ")
        append("\"requiredFamilyAdditions\": ${jsonStringArray(conformance.requiredFamilyAdditions.sorted())}, ")
        append("\"requiredFamilyOmissions\": ${jsonStringArray(conformance.requiredFamilyOmissions.sorted())}, ")
        append("\"forbiddenFamilyAdditions\": ${jsonStringArray(conformance.forbiddenFamilyAdditions.sorted())}, ")
        append("\"forbiddenFamilyOmissions\": ${jsonStringArray(conformance.forbiddenFamilyOmissions.sorted())}, ")
        append("\"requiredAliasAdditions\": ${jsonStringArray(conformance.requiredAliasAdditions.sorted())}, ")
        append("\"requiredAliasOmissions\": ${jsonStringArray(conformance.requiredAliasOmissions.sorted())}, ")
        append("\"forbiddenAliasAdditions\": ${jsonStringArray(conformance.forbiddenAliasAdditions.sorted())}, ")
        append("\"forbiddenAliasOmissions\": ${jsonStringArray(conformance.forbiddenAliasOmissions.sorted())}")
        append("}")
    }
}

fun modelFamilyFixtureLockEntries(): Map<String, String> {
    val snapshot = modelFamilyFixtureLockSnapshot()
    return linkedMapOf(
        "fixtureRequiredSelectors" to snapshot.requiredSelectors.joinToString(","),
        "fixtureRequiredFamilies" to snapshot.requiredFamilies.joinToString(","),
        "fixturePassed" to snapshot.passed.toString(),
        "fixtureRequiredFingerprint" to snapshot.requiredFingerprint,
        "fixtureInventoryFingerprint" to snapshot.inventoryFingerprint,
        "fixtureRequiredContentFingerprints" to modelFamilyFixtureFingerprintMapValue(
            snapshot.requiredContentFingerprints
        ),
        "fixturePresentContentFingerprints" to modelFamilyFixtureFingerprintMapValue(
            snapshot.presentContentFingerprints
        ),
        "fixtureAvailableFamilyCount" to snapshot.availableFamilyCount.toString(),
        "fixtureFamilyCount" to snapshot.fixtureFamilyCount.toString(),
        "fixtureRequiredFamilyCount" to snapshot.requiredFamilyCount.toString(),
        "fixtureRequiredPassedCount" to snapshot.requiredPassedCount.toString(),
        "fixtureMissingRequiredCount" to snapshot.missingRequiredCount.toString(),
        "fixtureProblemFamilyCount" to snapshot.problemFamilyCount.toString(),
        "fixtureMissingRequiredFamilies" to snapshot.missingRequiredFamilies.joinToString(","),
        "fixtureProblemFamilies" to snapshot.problemFamilies.joinToString(",")
    )
}

fun modelFamilyBundleLockEntries(selected: List<ModelFamilyModule>): Map<String, String> {
    val missingRequired = modelFamilyPolicyMissingRequired()
    val selectedForbidden = modelFamilyPolicySelectedForbidden()
    val missingRequiredAliases = modelFamilyPolicyMissingRequiredAliases()
    val selectedForbiddenAliases = modelFamilyPolicySelectedForbiddenAliases()
    val conformance = modelFamilyBundlePresetConformanceValidation()
    val selectedFamilyIds = selected.map { it.id }.toSet()
    val pendingTokenizerFamilyIds = selectedFamilyIds.intersect(pendingModelFamilyTokenizerMetadata)
    val productionTokenizerMetadataRequired = requestedModelFamilyBundlePreset?.id?.startsWith("prod_") == true
    val productionTokenizerMetadataReady = pendingTokenizerFamilyIds.isEmpty()
    val productionSafetyPassed = !productionTokenizerMetadataRequired || productionTokenizerMetadataReady
    val pendingTokenizerReasons = pendingTokenizerFamilyIds.associateWith(::modelFamilyTokenizerMetadataPendingReason)
    val policyViolationCount = modelFamilyPolicyViolationCount(
        missingRequired,
        selectedForbidden,
        missingRequiredAliases,
        selectedForbiddenAliases
    )
    return linkedMapOf(
        "lockSchemaVersion" to modelFamilyBundleLockSchemaVersion,
        "bundleSchemaVersion" to modelFamilyBundleManifestSchemaVersion,
        "bundlePreset" to (requestedModelFamilyBundlePreset?.id ?: ""),
        "bundleFingerprint" to modelFamilyBundleFingerprint(selected),
        "selectors" to requestedModelFamilyTokens.sorted().joinToString(","),
        "families" to selected.joinToString(",") { it.id },
        "familyCount" to selected.size.toString(),
        "profiles" to selected.map { it.profile }.distinct().sorted().joinToString(","),
        "presetConformanceStatus" to conformance.statusLabel,
        "presetConformanceMatchesPreset" to conformance.matchesPreset.toString(),
        "presetConformanceCleanPresetBuild" to conformance.cleanPresetBuild.toString(),
        "presetConformanceSelectorsMatch" to conformance.selectorsMatch.toString(),
        "presetConformancePolicyInputsMatch" to conformance.policyInputsMatch.toString(),
        "presetConformanceExplicitSelectorOverride" to conformance.explicitSelectorOverride.toString(),
        "presetConformanceExplicitPolicyOverride" to conformance.explicitPolicyOverride.toString(),
        "presetConformanceSelectorAdditions" to conformance.selectorAdditions.sorted().joinToString(","),
        "presetConformanceSelectorOmissions" to conformance.selectorOmissions.sorted().joinToString(","),
        "presetConformanceRequiredFamilyAdditions" to conformance.requiredFamilyAdditions.sorted().joinToString(","),
        "presetConformanceRequiredFamilyOmissions" to conformance.requiredFamilyOmissions.sorted().joinToString(","),
        "presetConformanceForbiddenFamilyAdditions" to conformance.forbiddenFamilyAdditions.sorted().joinToString(","),
        "presetConformanceForbiddenFamilyOmissions" to conformance.forbiddenFamilyOmissions.sorted().joinToString(","),
        "presetConformanceRequiredAliasAdditions" to conformance.requiredAliasAdditions.sorted().joinToString(","),
        "presetConformanceRequiredAliasOmissions" to conformance.requiredAliasOmissions.sorted().joinToString(","),
        "presetConformanceForbiddenAliasAdditions" to conformance.forbiddenAliasAdditions.sorted().joinToString(","),
        "presetConformanceForbiddenAliasOmissions" to conformance.forbiddenAliasOmissions.sorted().joinToString(","),
        "policySource" to modelFamilyPolicySource,
        "presetRequiredFamilies" to presetRequiredModelFamilyTokens.sorted().joinToString(","),
        "presetForbiddenFamilies" to presetForbiddenModelFamilyTokens.sorted().joinToString(","),
        "presetRequiredAliases" to presetRequiredModelFamilyAliasTokens.sorted().joinToString(","),
        "presetForbiddenAliases" to presetForbiddenModelFamilyAliasTokens.sorted().joinToString(","),
        "explicitRequiredFamilies" to explicitRequiredModelFamilyTokens.sorted().joinToString(","),
        "explicitForbiddenFamilies" to explicitForbiddenModelFamilyTokens.sorted().joinToString(","),
        "explicitRequiredAliases" to explicitRequiredModelFamilyAliasTokens.sorted().joinToString(","),
        "explicitForbiddenAliases" to explicitForbiddenModelFamilyAliasTokens.sorted().joinToString(","),
        "requiresDirectSafetensorRuntime" to requiresDirectSafetensorRuntime.toString(),
        "productionTokenizerMetadataRequired" to productionTokenizerMetadataRequired.toString(),
        "productionTokenizerMetadataReady" to productionTokenizerMetadataReady.toString(),
        "productionSafetyPassed" to productionSafetyPassed.toString(),
        "pendingTokenizerFamilies" to pendingTokenizerFamilyIds.sorted().joinToString(","),
        "pendingTokenizerReasons" to modelFamilyTokenizerPendingReasonsValue(pendingTokenizerReasons),
        "requiredFamilies" to requiredModelFamilies.sorted().joinToString(","),
        "forbiddenFamilies" to forbiddenModelFamilies.sorted().joinToString(","),
        "requiredAliases" to requiredModelFamilyAliasTokens.sorted().joinToString(","),
        "forbiddenAliases" to forbiddenModelFamilyAliasTokens.sorted().joinToString(","),
        "policyPassed" to (policyViolationCount == 0).toString(),
        "policyViolationCount" to policyViolationCount.toString()
    ).apply {
        putAll(modelFamilyFixtureLockEntries())
    }
}

fun modelFamilyBundleLockText(selected: List<ModelFamilyModule>): String {
    return buildString {
        appendLine("# Generated by :ui:gollek-cli:writeModelFamilyBundleLock")
        appendLine("# Regenerate only after intentionally changing the production model-family bundle.")
        modelFamilyBundleLockEntries(selected).forEach { (key, value) ->
            appendLine("$key=$value")
        }
    }
}

fun modelFamilyBundleLockInput(selected: List<ModelFamilyModule>): String {
    return modelFamilyBundleLockEntries(selected)
        .entries
        .joinToString("\n") { (key, value) -> "$key=$value" }
}

fun displayModelFamilyBundleLockValue(value: String?): String {
    return if (value == null) {
        "missing"
    } else {
        value.ifBlank { "none" }
    }
}

fun isModelFamilyBundleLockConformanceKey(key: String): Boolean {
    return key == "lockSchemaVersion" || key.startsWith("presetConformance")
}

fun isModelFamilyBundleLockFixtureKey(key: String): Boolean {
    return key == "lockSchemaVersion" || key.startsWith("fixture")
}

fun isModelFamilyBundleLockProductionSafetyKey(key: String): Boolean {
    return key == "productionTokenizerMetadataRequired" ||
            key == "productionTokenizerMetadataReady" ||
            key == "productionSafetyPassed" ||
            key == "pendingTokenizerFamilies" ||
            key == "pendingTokenizerReasons"
}

fun modelFamilyBundleLockDriftEntries(
    expected: Map<String, String>,
    actual: Properties
): List<ModelFamilyBundleLockDriftEntry> {
    return expected.entries.mapNotNull { (key, expectedValue) ->
        val actualValue = actual.getProperty(key)
        when {
            actualValue == null -> ModelFamilyBundleLockDriftEntry(
                key = key,
                expected = expectedValue,
                actual = null,
                status = "missing",
                conformanceRelated = isModelFamilyBundleLockConformanceKey(key),
                fixtureRelated = isModelFamilyBundleLockFixtureKey(key)
            )
            actualValue != expectedValue -> ModelFamilyBundleLockDriftEntry(
                key = key,
                expected = expectedValue,
                actual = actualValue,
                status = "changed",
                conformanceRelated = isModelFamilyBundleLockConformanceKey(key),
                fixtureRelated = isModelFamilyBundleLockFixtureKey(key)
            )
            else -> null
        }
    }
}

fun modelFamilyFixtureFingerprintLockDriftEntries(
    expected: Map<String, String>,
    actual: Properties
): List<ModelFamilyBundleLockDriftEntry> {
    return expected.entries.mapNotNull { (key, expectedValue) ->
        val actualValue = actual.getProperty(key)
        when {
            actualValue == null -> ModelFamilyBundleLockDriftEntry(
                key = key,
                expected = expectedValue,
                actual = null,
                status = "missing",
                conformanceRelated = false,
                fixtureRelated = true
            )
            actualValue != expectedValue -> ModelFamilyBundleLockDriftEntry(
                key = key,
                expected = expectedValue,
                actual = actualValue,
                status = "changed",
                conformanceRelated = false,
                fixtureRelated = true
            )
            else -> null
        }
    }
}

fun modelFamilyBundleLockKeyProblems(entries: List<ModelFamilyBundleLockDriftEntry>): List<String> {
    return entries.map { entry ->
        if (entry.actual == null) {
            "${entry.key} is missing, expected '${entry.expected}'"
        } else {
            "${entry.key} is '${entry.actual}', expected '${entry.expected}'"
        }
    }
}

fun modelFamilyBundleLockDriftSummaries(expected: Map<String, String>, actual: Properties): List<String> {
    val summaries = mutableListOf<String>()

    fun summarizeKey(key: String, label: String) {
        val expectedValue = expected[key]
        val actualValue = actual.getProperty(key)
        if (expectedValue != null && actualValue != expectedValue) {
            summaries += "$label changed from '${displayModelFamilyBundleLockValue(actualValue)}' " +
                    "to '${displayModelFamilyBundleLockValue(expectedValue)}'"
        }
    }

    summarizeKey("lockSchemaVersion", "lock schema")
    summarizeKey("presetConformanceStatus", "preset conformance status")
    summarizeKey("presetConformanceCleanPresetBuild", "clean preset build")
    summarizeKey("presetConformanceSelectorsMatch", "selector conformance")
    summarizeKey("presetConformancePolicyInputsMatch", "policy-input conformance")
    summarizeKey("presetConformanceExplicitSelectorOverride", "explicit selector override")
    summarizeKey("presetConformanceExplicitPolicyOverride", "explicit policy override")
    summarizeKey("presetConformanceSelectorAdditions", "selector additions")
    summarizeKey("presetConformanceSelectorOmissions", "selector omissions")
    summarizeKey("presetConformanceRequiredFamilyAdditions", "required family additions")
    summarizeKey("presetConformanceRequiredFamilyOmissions", "required family omissions")
    summarizeKey("presetConformanceForbiddenFamilyAdditions", "forbidden family additions")
    summarizeKey("presetConformanceForbiddenFamilyOmissions", "forbidden family omissions")
    summarizeKey("presetConformanceRequiredAliasAdditions", "required alias additions")
    summarizeKey("presetConformanceRequiredAliasOmissions", "required alias omissions")
    summarizeKey("presetConformanceForbiddenAliasAdditions", "forbidden alias additions")
    summarizeKey("presetConformanceForbiddenAliasOmissions", "forbidden alias omissions")
    summarizeKey("productionTokenizerMetadataRequired", "production tokenizer metadata requirement")
    summarizeKey("productionTokenizerMetadataReady", "production tokenizer metadata readiness")
    summarizeKey("productionSafetyPassed", "production tokenizer safety gate")
    summarizeKey("pendingTokenizerReasons", "pending tokenizer reasons")

    return summaries
}

fun modelFamilyBundleLockListValue(value: String?): Set<String> {
    if (value.isNullOrBlank()) {
        return emptySet()
    }
    return value.split(",")
        .map { it.trim() }
        .filter { it.isNotBlank() }
        .toSet()
}

fun modelFamilyBundleLockSelectionDriftEntries(
    expected: Map<String, String>,
    actual: Properties
): List<ModelFamilyBundleSelectionDriftEntry> {
    val listKeys = listOf(
        "selectors" to "selector",
        "families" to "family",
        "profiles" to "profile",
        "pendingTokenizerFamilies" to "pendingTokenizerFamily",
        "requiredFamilies" to "requiredFamily",
        "forbiddenFamilies" to "forbiddenFamily",
        "requiredAliases" to "requiredAlias",
        "forbiddenAliases" to "forbiddenAlias"
    )
    return listKeys.flatMap { (key, scope) ->
        val expectedValue = expected[key] ?: return@flatMap emptyList()
        val actualValue = actual.getProperty(key)
        if (actualValue == expectedValue) {
            return@flatMap emptyList()
        }

        val expectedItems = modelFamilyBundleLockListValue(expectedValue)
        val actualItems = modelFamilyBundleLockListValue(actualValue)
        val added = (expectedItems - actualItems)
            .sorted()
            .map { value ->
                ModelFamilyBundleSelectionDriftEntry(
                    key = key,
                    scope = scope,
                    value = value,
                    status = "added"
                )
            }
        val removed = (actualItems - expectedItems)
            .sorted()
            .map { value ->
                ModelFamilyBundleSelectionDriftEntry(
                    key = key,
                    scope = scope,
                    value = value,
                    status = "removed"
                )
            }
        added + removed
    }
}

fun modelFamilyBundleLockSelectionDriftSummaries(
    expected: Map<String, String>,
    actual: Properties
): List<String> {
    return modelFamilyBundleLockSelectionDriftEntries(expected, actual)
        .groupBy { it.key }
        .map { (key, entries) ->
            val label = when (key) {
                "selectors" -> "selectors"
                "families" -> "selected families"
                "profiles" -> "selected profiles"
                "pendingTokenizerFamilies" -> "pending tokenizer families"
                "requiredFamilies" -> "required family policy"
                "forbiddenFamilies" -> "forbidden family policy"
                "requiredAliases" -> "required alias policy"
                "forbiddenAliases" -> "forbidden alias policy"
                else -> key
            }
            val added = entries.filter { it.status == "added" }.map { it.value }.sorted()
            val removed = entries.filter { it.status == "removed" }.map { it.value }.sorted()
            val detail = listOfNotNull(
                added.takeIf { it.isNotEmpty() }?.let { "added ${it.joinToString(", ")}" },
                removed.takeIf { it.isNotEmpty() }?.let { "removed ${it.joinToString(", ")}" }
            ).joinToString("; ")
            if (detail.isBlank()) {
                "$label changed"
            } else {
                "$label changed ($detail)"
            }
        }
        .sorted()
}

fun modelFamilyBundleLockProductionSafetyDriftSummaries(
    expected: Map<String, String>,
    actual: Properties
): List<String> {
    val summaries = mutableListOf<String>()

    fun summarizeKey(key: String, label: String) {
        val expectedValue = expected[key]
        val actualValue = actual.getProperty(key)
        if (expectedValue != null && actualValue != expectedValue) {
            summaries += "$label changed from '${displayModelFamilyBundleLockValue(actualValue)}' " +
                    "to '${displayModelFamilyBundleLockValue(expectedValue)}'"
        }
    }

    summarizeKey("productionTokenizerMetadataRequired", "production tokenizer metadata requirement")
    summarizeKey("productionTokenizerMetadataReady", "production tokenizer metadata readiness")
    summarizeKey("productionSafetyPassed", "production tokenizer safety gate")
    summarizeKey("pendingTokenizerReasons", "pending tokenizer reasons")

    val pendingFamilyExpected = expected["pendingTokenizerFamilies"]
    val pendingFamilyActual = actual.getProperty("pendingTokenizerFamilies")
    if (pendingFamilyExpected != null && pendingFamilyActual != pendingFamilyExpected) {
        val expectedFamilies = modelFamilyBundleLockListValue(pendingFamilyExpected)
        val actualFamilies = modelFamilyBundleLockListValue(pendingFamilyActual)
        val added = (expectedFamilies - actualFamilies).sorted()
        val removed = (actualFamilies - expectedFamilies).sorted()
        val detail = listOfNotNull(
            added.takeIf { it.isNotEmpty() }?.let { "added ${it.joinToString(", ")}" },
            removed.takeIf { it.isNotEmpty() }?.let { "removed ${it.joinToString(", ")}" }
        ).joinToString("; ")
        summaries += if (detail.isBlank()) {
            "pending tokenizer families changed"
        } else {
            "pending tokenizer families changed ($detail)"
        }
    }

    return summaries.sorted()
}

fun modelFamilyBundleLockScalarDriftEntries(
    expected: Map<String, String>,
    actual: Properties
): List<ModelFamilyBundleScalarDriftEntry> {
    val scalarKeys = listOf(
        Triple("lockSchemaVersion", "schema", "lock schema version"),
        Triple("bundleSchemaVersion", "schema", "bundle schema version"),
        Triple("bundlePreset", "bundle", "bundle preset"),
        Triple("bundleFingerprint", "bundle", "bundle fingerprint"),
        Triple("familyCount", "bundle", "family count"),
        Triple("policySource", "policy", "policy source"),
        Triple("policyPassed", "policy", "policy gate"),
        Triple("policyViolationCount", "policy", "policy violation count"),
        Triple("productionTokenizerMetadataRequired", "production", "production tokenizer metadata requirement"),
        Triple("productionTokenizerMetadataReady", "production", "production tokenizer metadata readiness"),
        Triple("productionSafetyPassed", "production", "production tokenizer safety gate"),
        Triple("pendingTokenizerReasons", "production", "pending tokenizer reasons"),
        Triple("presetConformanceStatus", "conformance", "preset conformance status"),
        Triple("presetConformanceMatchesPreset", "conformance", "preset match"),
        Triple("presetConformanceCleanPresetBuild", "conformance", "clean preset build"),
        Triple("presetConformanceSelectorsMatch", "conformance", "selector conformance"),
        Triple("presetConformancePolicyInputsMatch", "conformance", "policy-input conformance"),
        Triple("presetConformanceExplicitSelectorOverride", "conformance", "explicit selector override"),
        Triple("presetConformanceExplicitPolicyOverride", "conformance", "explicit policy override"),
        Triple("fixturePassed", "fixture", "fixture gate"),
        Triple("fixtureRequiredFingerprint", "fixture", "required fixture fingerprint"),
        Triple("fixtureInventoryFingerprint", "fixture", "fixture inventory fingerprint"),
        Triple("fixtureAvailableFamilyCount", "fixture", "available fixture family count"),
        Triple("fixtureFamilyCount", "fixture", "fixture family count"),
        Triple("fixtureRequiredFamilyCount", "fixture", "required fixture family count"),
        Triple("fixtureRequiredPassedCount", "fixture", "required fixture passed count"),
        Triple("fixtureMissingRequiredCount", "fixture", "missing required fixture count"),
        Triple("fixtureProblemFamilyCount", "fixture", "problem fixture family count")
    )
    return scalarKeys.mapNotNull { (key, scope, label) ->
        val expectedValue = expected[key] ?: return@mapNotNull null
        val actualValue = actual.getProperty(key)
        when {
            actualValue == null -> ModelFamilyBundleScalarDriftEntry(
                key = key,
                scope = scope,
                label = label,
                status = "missing",
                expected = expectedValue,
                actual = null
            )
            actualValue != expectedValue -> ModelFamilyBundleScalarDriftEntry(
                key = key,
                scope = scope,
                label = label,
                status = "changed",
                expected = expectedValue,
                actual = actualValue
            )
            else -> null
        }
    }
}

fun modelFamilyBundleLockFingerprintMap(value: String?): Map<String, String> {
    if (value.isNullOrBlank()) {
        return emptyMap()
    }
    return value.split(",")
        .mapNotNull { entry ->
            val separator = entry.indexOf("=")
            if (separator <= 0) {
                null
            } else {
                entry.substring(0, separator) to entry.substring(separator + 1)
            }
        }
        .toMap()
}

fun modelFamilyBundleLockFixtureFingerprintDriftEntries(
    expected: Map<String, String>,
    actual: Properties
): List<ModelFamilyFixtureFingerprintDriftEntry> {
    val fingerprintKeys = listOf(
        "fixtureRequiredContentFingerprints" to "required",
        "fixturePresentContentFingerprints" to "present"
    )
    return fingerprintKeys.flatMap { (key, scope) ->
        val expectedValue = expected[key] ?: return@flatMap emptyList()
        val actualValue = actual.getProperty(key)
        if (actualValue == expectedValue) {
            return@flatMap emptyList()
        }

        val expectedMap = modelFamilyBundleLockFingerprintMap(expectedValue)
        val actualMap = modelFamilyBundleLockFingerprintMap(actualValue)
        val changed = expectedMap.keys
            .intersect(actualMap.keys)
            .filter { familyId -> expectedMap[familyId] != actualMap[familyId] }
            .sorted()
            .map { familyId ->
                ModelFamilyFixtureFingerprintDriftEntry(
                    key = key,
                    scope = scope,
                    familyId = familyId,
                    status = "changed",
                    expected = expectedMap[familyId],
                    actual = actualMap[familyId]
                )
            }
        val added = (expectedMap.keys - actualMap.keys)
            .sorted()
            .map { familyId ->
                ModelFamilyFixtureFingerprintDriftEntry(
                    key = key,
                    scope = scope,
                    familyId = familyId,
                    status = "added",
                    expected = expectedMap[familyId],
                    actual = null
                )
            }
        val removed = (actualMap.keys - expectedMap.keys)
            .sorted()
            .map { familyId ->
                ModelFamilyFixtureFingerprintDriftEntry(
                    key = key,
                    scope = scope,
                    familyId = familyId,
                    status = "removed",
                    expected = null,
                    actual = actualMap[familyId]
                )
            }
        changed + added + removed
    }
}

fun modelFamilyBundleLockFixtureDriftSummaries(expected: Map<String, String>, actual: Properties): List<String> {
    val summaries = mutableListOf<String>()

    fun summarizeKey(key: String, label: String) {
        val expectedValue = expected[key]
        val actualValue = actual.getProperty(key)
        if (expectedValue != null && actualValue != expectedValue) {
            summaries += "$label changed from '${displayModelFamilyBundleLockValue(actualValue)}' " +
                    "to '${displayModelFamilyBundleLockValue(expectedValue)}'"
        }
    }

    fun summarizeFingerprintMap(key: String, label: String) {
        val expectedValue = expected[key]
        val actualValue = actual.getProperty(key)
        if (expectedValue == null || actualValue == expectedValue) {
            return
        }
        if (actualValue == null) {
            summaries += "$label changed from 'missing' to '${displayModelFamilyBundleLockValue(expectedValue)}'"
            return
        }

        val expectedMap = modelFamilyBundleLockFingerprintMap(expectedValue)
        val actualMap = modelFamilyBundleLockFingerprintMap(actualValue)
        val added = (expectedMap.keys - actualMap.keys).sorted()
        val removed = (actualMap.keys - expectedMap.keys).sorted()
        val changed = expectedMap.keys
            .intersect(actualMap.keys)
            .filter { familyId -> expectedMap[familyId] != actualMap[familyId] }
            .sorted()
        val detail = listOfNotNull(
            changed.takeIf { it.isNotEmpty() }?.let { "changed ${it.joinToString(", ")}" },
            added.takeIf { it.isNotEmpty() }?.let { "added ${it.joinToString(", ")}" },
            removed.takeIf { it.isNotEmpty() }?.let { "removed ${it.joinToString(", ")}" }
        ).joinToString("; ")

        summaries += if (detail.isBlank()) {
            "$label changed"
        } else {
            "$label changed ($detail)"
        }
    }

    summarizeKey("lockSchemaVersion", "lock schema")
    summarizeKey("fixtureRequiredSelectors", "required fixture selectors")
    summarizeKey("fixtureRequiredFamilies", "required fixture families")
    summarizeKey("fixturePassed", "fixture gate")
    summarizeKey("fixtureRequiredFingerprint", "required fixture fingerprint")
    summarizeKey("fixtureInventoryFingerprint", "fixture inventory fingerprint")
    summarizeFingerprintMap("fixtureRequiredContentFingerprints", "required fixture content fingerprints")
    summarizeFingerprintMap("fixturePresentContentFingerprints", "present fixture content fingerprints")
    summarizeKey("fixtureAvailableFamilyCount", "available fixture family count")
    summarizeKey("fixtureFamilyCount", "fixture family count")
    summarizeKey("fixtureRequiredFamilyCount", "required fixture family count")
    summarizeKey("fixtureRequiredPassedCount", "required fixture passed count")
    summarizeKey("fixtureMissingRequiredCount", "missing required fixture count")
    summarizeKey("fixtureProblemFamilyCount", "problem fixture family count")
    summarizeKey("fixtureMissingRequiredFamilies", "missing required fixture families")
    summarizeKey("fixtureProblemFamilies", "problem fixture families")

    return summaries
}

fun jsonModelFamilyBundleLockDriftEntries(entries: List<ModelFamilyBundleLockDriftEntry>): String {
    return entries.joinToString(prefix = "[", postfix = "]") { entry ->
        buildString {
            append("{")
            append("\"key\": ${jsonString(entry.key)}, ")
            append("\"status\": ${jsonString(entry.status)}, ")
            append("\"expected\": ${jsonString(entry.expected)}, ")
            append("\"actual\": ${entry.actual?.let(::jsonString) ?: "null"}, ")
            append("\"conformanceRelated\": ${entry.conformanceRelated}, ")
            append("\"fixtureRelated\": ${entry.fixtureRelated}")
            append("}")
        }
    }
}

fun jsonModelFamilyBundleSelectionDriftEntries(entries: List<ModelFamilyBundleSelectionDriftEntry>): String {
    return entries.joinToString(prefix = "[", postfix = "]") { entry ->
        buildString {
            append("{")
            append("\"key\": ${jsonString(entry.key)}, ")
            append("\"scope\": ${jsonString(entry.scope)}, ")
            append("\"value\": ${jsonString(entry.value)}, ")
            append("\"status\": ${jsonString(entry.status)}")
            append("}")
        }
    }
}

fun jsonModelFamilyBundleScalarDriftEntries(entries: List<ModelFamilyBundleScalarDriftEntry>): String {
    return entries.joinToString(prefix = "[", postfix = "]") { entry ->
        buildString {
            append("{")
            append("\"key\": ${jsonString(entry.key)}, ")
            append("\"scope\": ${jsonString(entry.scope)}, ")
            append("\"label\": ${jsonString(entry.label)}, ")
            append("\"status\": ${jsonString(entry.status)}, ")
            append("\"expected\": ${jsonString(entry.expected)}, ")
            append("\"actual\": ${entry.actual?.let(::jsonString) ?: "null"}")
            append("}")
        }
    }
}

fun jsonModelFamilyFixtureFingerprintDriftEntries(
    entries: List<ModelFamilyFixtureFingerprintDriftEntry>
): String {
    return entries.joinToString(prefix = "[", postfix = "]") { entry ->
        buildString {
            append("{")
            append("\"key\": ${jsonString(entry.key)}, ")
            append("\"scope\": ${jsonString(entry.scope)}, ")
            append("\"familyId\": ${jsonString(entry.familyId)}, ")
            append("\"status\": ${jsonString(entry.status)}, ")
            append("\"expected\": ${entry.expected?.let(::jsonString) ?: "null"}, ")
            append("\"actual\": ${entry.actual?.let(::jsonString) ?: "null"}")
            append("}")
        }
    }
}

fun modelFamilyBundleLockDriftReportJson(
    lockFile: File,
    expected: Map<String, String>,
    actual: Properties?,
    missingLock: Boolean
): String {
    val entries = if (actual == null) {
        emptyList()
    } else {
        modelFamilyBundleLockDriftEntries(expected, actual)
    }
    val conformanceSummaries = if (actual == null) {
        emptyList()
    } else {
        modelFamilyBundleLockDriftSummaries(expected, actual)
    }
    val selectionSummaries = if (actual == null) {
        emptyList()
    } else {
        modelFamilyBundleLockSelectionDriftSummaries(expected, actual)
    }
    val productionSafetySummaries = if (actual == null) {
        emptyList()
    } else {
        modelFamilyBundleLockProductionSafetyDriftSummaries(expected, actual)
    }
    val bundleSelectionDrift = if (actual == null) {
        emptyList()
    } else {
        modelFamilyBundleLockSelectionDriftEntries(expected, actual)
    }
    val productionSafetyDrift = entries.filter {
        isModelFamilyBundleLockProductionSafetyKey(it.key)
    }
    val bundleScalarDrift = if (actual == null) {
        emptyList()
    } else {
        modelFamilyBundleLockScalarDriftEntries(expected, actual)
    }
    val fixtureSummaries = if (actual == null) {
        emptyList()
    } else {
        modelFamilyBundleLockFixtureDriftSummaries(expected, actual)
    }
    val fixtureFingerprintDrift = if (actual == null) {
        emptyList()
    } else {
        modelFamilyBundleLockFixtureFingerprintDriftEntries(expected, actual)
    }
    return buildString {
        appendLine("{")
        appendLine("  \"schemaVersion\": 1,")
        appendLine("  \"lockPath\": ${jsonString(lockFile.relativeTo(projectDir).path)},")
        appendLine("  \"missingLock\": $missingLock,")
        appendLine("  \"passed\": ${!missingLock && entries.isEmpty()},")
        appendLine("  \"driftCount\": ${entries.size},")
        appendLine("  \"bundleSelectionDriftCount\": ${bundleSelectionDrift.size},")
        appendLine("  \"bundleScalarDriftCount\": ${bundleScalarDrift.size},")
        appendLine("  \"productionSafetyDriftCount\": ${productionSafetyDrift.size},")
        appendLine("  \"conformanceDriftCount\": ${entries.count { it.conformanceRelated }},")
        appendLine("  \"fixtureDriftCount\": ${entries.count { it.fixtureRelated }},")
        appendLine("  \"fixtureFingerprintDriftCount\": ${fixtureFingerprintDrift.size},")
        appendLine("  \"conformanceSummaries\": ${jsonStringArray(conformanceSummaries)},")
        appendLine("  \"selectionSummaries\": ${jsonStringArray(selectionSummaries)},")
        appendLine("  \"productionSafetySummaries\": ${jsonStringArray(productionSafetySummaries)},")
        appendLine("  \"productionSafetyDrift\": ${jsonModelFamilyBundleLockDriftEntries(productionSafetyDrift)},")
        appendLine("  \"bundleSelectionDrift\": ${jsonModelFamilyBundleSelectionDriftEntries(bundleSelectionDrift)},")
        appendLine("  \"bundleScalarDrift\": ${jsonModelFamilyBundleScalarDriftEntries(bundleScalarDrift)},")
        appendLine("  \"fixtureSummaries\": ${jsonStringArray(fixtureSummaries)},")
        appendLine("  \"fixtureFingerprintDrift\": ${jsonModelFamilyFixtureFingerprintDriftEntries(fixtureFingerprintDrift)},")
        appendLine("  \"expectedPresetConformanceStatus\": ${jsonString(expected["presetConformanceStatus"] ?: "")},")
        appendLine(
            "  \"actualPresetConformanceStatus\": ${
                actual?.getProperty("presetConformanceStatus")?.let(::jsonString) ?: "null"
            },"
        )
        appendLine("  \"expectedFixtureStatus\": ${jsonString(expected["fixturePassed"] ?: "")},")
        appendLine(
            "  \"actualFixtureStatus\": ${
                actual?.getProperty("fixturePassed")?.let(::jsonString) ?: "null"
            },"
        )
        appendLine("  \"currentBundleStatus\": ${jsonModelFamilyBundleLockCurrentStatus(selectedModelFamilyModules())},")
        appendLine("  \"drift\": ${jsonModelFamilyBundleLockDriftEntries(entries)}")
        appendLine("}")
    }
}

fun jsonModelFamilyBundleLockCurrentStatus(selected: List<ModelFamilyModule>): String {
    val selectedIds = selected.map { it.id }
    val selectedFamilyIds = selectedIds.toSet()
    val pendingTokenizerFamilyIds = selectedFamilyIds.intersect(pendingModelFamilyTokenizerMetadata).sorted()
    val pendingTokenizerReasons = pendingTokenizerFamilyIds.associateWith(::modelFamilyTokenizerMetadataPendingReason)
    val productionTokenizerMetadataRequired = requestedModelFamilyBundlePreset?.id?.startsWith("prod_") == true
    val productionTokenizerMetadataReady = pendingTokenizerFamilyIds.isEmpty()
    val productionSafetyPassed = !productionTokenizerMetadataRequired || productionTokenizerMetadataReady
    val productionSafetySummary = when {
        !productionTokenizerMetadataRequired -> "not required"
        productionTokenizerMetadataReady -> "passed (tokenizer metadata ready)"
        else -> "failed (${pendingTokenizerFamilyIds.size} pending tokenizer family(s): " +
                pendingTokenizerFamilyIds.joinToString(", ") + ")"
    }
    val missingRequired = modelFamilyPolicyMissingRequired()
    val selectedForbidden = modelFamilyPolicySelectedForbidden()
    val missingRequiredAliases = modelFamilyPolicyMissingRequiredAliases()
    val selectedForbiddenAliases = modelFamilyPolicySelectedForbiddenAliases()
    val policyViolationCount = modelFamilyPolicyViolationCount(
        missingRequired,
        selectedForbidden,
        missingRequiredAliases,
        selectedForbiddenAliases
    )
    return buildString {
        append("{")
        append("\"schemaVersion\": ${modelFamilyBundleManifestSchemaVersion.toIntOrNull() ?: 0}, ")
        append("\"bundleFingerprint\": ${jsonString(modelFamilyBundleFingerprint(selected))}, ")
        append("\"bundlePreset\": ${requestedModelFamilyBundlePreset?.id?.let(::jsonString) ?: "null"}, ")
        append("\"activeBundlePresetConformance\": ${jsonModelFamilyBundlePresetConformance()}, ")
        append("\"selectorSource\": ${jsonString(requestedModelFamilySelectorSource)}, ")
        append("\"policySource\": ${jsonString(modelFamilyPolicySource)}, ")
        append("\"selectors\": ${jsonStringArray(requestedModelFamilyTokens.sorted())}, ")
        append("\"detached\": ${selected.isEmpty()}, ")
        append("\"familyCount\": ${selected.size}, ")
        append("\"families\": ${jsonStringArray(selectedIds)}, ")
        append("\"profiles\": ${jsonStringArray(selected.map { it.profile }.distinct().sorted())}, ")
        append("\"productionSafety\": {")
        append("\"tokenizerMetadataRequired\": $productionTokenizerMetadataRequired, ")
        append("\"tokenizerMetadataReady\": $productionTokenizerMetadataReady, ")
        append("\"passed\": $productionSafetyPassed, ")
        append("\"status\": ${jsonString(if (productionSafetyPassed) "passed" else "failed")}, ")
        append("\"summary\": ${jsonString(productionSafetySummary)}, ")
        append("\"pendingTokenizerFamilyCount\": ${pendingTokenizerFamilyIds.size}, ")
        append("\"pendingTokenizerFamilies\": ${jsonStringArray(pendingTokenizerFamilyIds)}, ")
        append("\"pendingTokenizerReasons\": ${jsonStringMap(pendingTokenizerReasons)}")
        append("}, ")
        append("\"policyStatus\": {")
        append("\"passed\": ${policyViolationCount == 0}, ")
        append("\"violationCount\": $policyViolationCount, ")
        append("\"missingRequired\": ${jsonStringArray(missingRequired.sorted())}, ")
        append("\"selectedForbidden\": ${jsonStringArray(selectedForbidden.sorted())}, ")
        append("\"missingRequiredAliasCount\": ${missingRequiredAliases.size}, ")
        append("\"selectedForbiddenAliasCount\": ${selectedForbiddenAliases.size}")
        append("}, ")
        append("\"fixtureStatus\": ${jsonModelFamilyFixtureStatus(modelFamilyFixtureInventories())}")
        append("}")
    }
}

fun modelFamilyFixtureFingerprintLockDriftReportJson(
    lockFile: File,
    expected: Map<String, String>,
    actual: Properties?,
    missingLock: Boolean
): String {
    val entries = if (actual == null) {
        emptyList()
    } else {
        modelFamilyFixtureFingerprintLockDriftEntries(expected, actual)
    }
    val fixtureSummaries = if (actual == null) {
        emptyList()
    } else {
        modelFamilyBundleLockFixtureDriftSummaries(expected, actual)
    }
    val fixtureFingerprintDrift = if (actual == null) {
        emptyList()
    } else {
        modelFamilyBundleLockFixtureFingerprintDriftEntries(expected, actual)
    }
    return buildString {
        appendLine("{")
        appendLine("  \"schemaVersion\": 1,")
        appendLine("  \"lockPath\": ${jsonString(lockFile.relativeTo(projectDir).path)},")
        appendLine("  \"missingLock\": $missingLock,")
        appendLine("  \"passed\": ${!missingLock && entries.isEmpty()},")
        appendLine("  \"driftCount\": ${entries.size},")
        appendLine("  \"fixtureDriftCount\": ${entries.count { it.fixtureRelated }},")
        appendLine("  \"fixtureFingerprintDriftCount\": ${fixtureFingerprintDrift.size},")
        appendLine("  \"fixtureSummaries\": ${jsonStringArray(fixtureSummaries)},")
        appendLine("  \"fixtureFingerprintDrift\": ${jsonModelFamilyFixtureFingerprintDriftEntries(fixtureFingerprintDrift)},")
        appendLine("  \"expectedFixtureStatus\": ${jsonString(expected["fixturePassed"] ?: "")},")
        appendLine(
            "  \"actualFixtureStatus\": ${
                actual?.getProperty("fixturePassed")?.let(::jsonString) ?: "null"
            },"
        )
        appendLine("  \"currentFingerprintStatus\": ${jsonModelFamilyFixtureFingerprintStatus()},")
        appendLine("  \"drift\": ${jsonModelFamilyBundleLockDriftEntries(entries)}")
        appendLine("}")
    }
}

fun modelFamilyBundlePresetInput(): String {
    return modelFamilyBundlePresets.sortedBy { it.id }
        .joinToString("|") { preset ->
            listOf(
                preset.id,
                preset.selectors.sorted().joinToString("+"),
                preset.requiredFamilies.sorted().joinToString("+"),
                preset.forbiddenFamilies.sorted().joinToString("+"),
                preset.requiredAliases.sorted().joinToString("+"),
                preset.forbiddenAliases.sorted().joinToString("+")
            ).joinToString(":")
        }
}

fun jsonModelFamilyModuleCatalogEntries(): String {
    return modelFamilyModules.sortedBy { it.id }
        .joinToString(prefix = "[", postfix = "]") { module ->
            val included = modelFamilyProject(module) != null
            val pluginDescriptorPresent = modelFamilyPluginDescriptorPresent(module.path)
            val modelFamilyServicePresent = modelFamilyServiceDescriptorPresent(module.path)
            val gollekPluginServicePresent = gollekPluginServiceDescriptorPresent(module.path)
            val serviceDescriptorComplete = modelFamilyServiceDescriptorsComplete(module.path)
            val tokenizerKinds = modelFamilyPluginDescriptorTokenizerKinds(module.path)
            val tokenizerMetadataPending = module.id in pendingModelFamilyTokenizerMetadata
            val tokenizerMetadataPendingReason = modelFamilyTokenizerMetadataPendingReason(module.id)
            val tokenizerReady = modelFamilyModuleTokenizerReady(module)
            val metadataOnly = modelFamilyModuleMetadataOnly(module)
            val directSafetensorReady = modelFamilyModuleDirectSafetensorReady(module)
            val productionReady = modelFamilyModuleProductionReady(module)
            val readiness = modelFamilyModuleReadiness(module)
            buildString {
                append("{")
                append("\"id\": ${jsonString(module.id)}, ")
                append("\"profile\": ${jsonString(module.profile)}, ")
                append("\"path\": ${jsonString(module.path)}, ")
                append("\"included\": $included, ")
                append("\"selected\": ${module.id in bundledModelFamilies}, ")
                append("\"pluginDescriptorPresent\": $pluginDescriptorPresent, ")
                append("\"modelFamilyServiceDescriptorPresent\": $modelFamilyServicePresent, ")
                append("\"gollekPluginServiceDescriptorPresent\": $gollekPluginServicePresent, ")
                append("\"serviceDescriptorComplete\": $serviceDescriptorComplete, ")
                append("\"tokenizerMetadataPresent\": ${tokenizerKinds.isNotEmpty()}, ")
                append("\"tokenizerMetadataPending\": $tokenizerMetadataPending, ")
                append("\"tokenizerMetadataPendingReason\": ${jsonString(tokenizerMetadataPendingReason)}, ")
                append("\"tokenizerKinds\": ${jsonStringArray(tokenizerKinds)}, ")
                append("\"tokenizerReady\": $tokenizerReady, ")
                append("\"metadataOnly\": $metadataOnly, ")
                append("\"directSafetensorReady\": $directSafetensorReady, ")
                append("\"productionReady\": $productionReady, ")
                append("\"readiness\": ${jsonString(readiness)}")
                append("}")
            }
        }
}

fun jsonIncludedModelFamilyProjectEntries(): String {
    val catalogPaths = catalogModelFamilyProjectPaths()
    val catalogByPath = modelFamilyModules.associateBy { it.path }
    val catalogedPluginPaths = includedCatalogedModelFamilyProjectPaths()
    return includedModelFamilyProjectPaths().sorted()
        .joinToString(prefix = "[", postfix = "]") { path ->
            val module = catalogByPath[path]
            val pluginDescriptorPresent = modelFamilyPluginDescriptorPresent(path)
            val modelFamilyServicePresent = modelFamilyServiceDescriptorPresent(path)
            val gollekPluginServicePresent = gollekPluginServiceDescriptorPresent(path)
            buildString {
                append("{")
                append("\"path\": ${jsonString(path)}, ")
                append("\"cataloged\": ${path in catalogPaths}, ")
                append("\"supportOnly\": ${path !in catalogedPluginPaths}, ")
                append("\"pluginDescriptorPresent\": $pluginDescriptorPresent, ")
                append("\"modelFamilyServiceDescriptorPresent\": $modelFamilyServicePresent, ")
                append("\"gollekPluginServiceDescriptorPresent\": $gollekPluginServicePresent, ")
                append("\"serviceDescriptorComplete\": ${modelFamilyServiceDescriptorsComplete(path)}")
                if (module != null) {
                    append(", \"id\": ${jsonString(module.id)}")
                    append(", \"profile\": ${jsonString(module.profile)}")
                    append(", \"selected\": ${module.id in bundledModelFamilies}")
                }
                append("}")
            }
        }
}

fun jsonModelFamilyModuleCatalogSummary(): String {
    val catalogPaths = catalogModelFamilyProjectPaths()
    val includedPaths = includedModelFamilyProjectPaths()
    val missingProjects = modelFamilyModuleCatalogMissingProjects()
    val uncatalogedProjects = modelFamilyModuleCatalogUncatalogedProjects()
    val supportOnlyProjects = modelFamilyModuleCatalogSupportOnlyProjects()
    val serviceIncompleteProjects = modelFamilyModuleCatalogServiceIncompleteProjects()
    val tokenizerMissingFamilies = modelFamilyModuleCatalogTokenizerMetadataMissingFamilies()
    val tokenizerPendingFamilies = modelFamilyModuleCatalogTokenizerMetadataPendingFamilies()
    val tokenizerPendingReasons = modelFamilyModuleCatalogTokenizerMetadataPendingReasons()
    val tokenizerReadyFamilies = modelFamilyModules
        .filter(::modelFamilyModuleTokenizerReady)
        .map { it.id }
        .toSet()
    val productionReadyFamilies = modelFamilyModules
        .filter(::modelFamilyModuleProductionReady)
        .map { it.id }
        .toSet()
    val directSafetensorReadyFamilies = modelFamilyModules
        .filter(::modelFamilyModuleDirectSafetensorReady)
        .map { it.id }
        .toSet()
    val productionReadinessPendingFamilies = modelFamilyModuleCatalogProductionReadinessPendingFamilies()
    val directSafetensorPendingFamilies = modelFamilyModuleCatalogDirectSafetensorPendingFamilies()
    val metadataOnlyFamilies = modelFamilyModules
        .filter(::modelFamilyModuleMetadataOnly)
        .map { it.id }
        .toSet()
    val readinessCounts = modelFamilyModules
        .groupingBy(::modelFamilyModuleReadiness)
        .eachCount()
    val catalogedIncludedCount = catalogPaths.intersect(includedPaths).size
    val passed = missingProjects.isEmpty() &&
            uncatalogedProjects.isEmpty() &&
            serviceIncompleteProjects.isEmpty() &&
            tokenizerMissingFamilies.isEmpty() &&
            productionReadinessPendingFamilies.isEmpty() &&
            directSafetensorPendingFamilies.isEmpty()
    return buildString {
        append("{")
        append("\"passed\": $passed, ")
        append("\"catalogFamilyCount\": ${modelFamilyModules.size}, ")
        append("\"catalogProjectCount\": ${catalogPaths.size}, ")
        append("\"includedModelProjectCount\": ${includedPaths.size}, ")
        append("\"includedCatalogedProjectCount\": $catalogedIncludedCount, ")
        append("\"supportOnlyProjectCount\": ${supportOnlyProjects.size}, ")
        append("\"serviceDescriptorCompleteProjectCount\": ${catalogedIncludedCount - serviceIncompleteProjects.size}, ")
        append("\"serviceDescriptorIncompleteProjectCount\": ${serviceIncompleteProjects.size}, ")
        append("\"tokenizerMetadataReadyFamilyCount\": ${tokenizerReadyFamilies.size}, ")
        append("\"tokenizerMetadataPendingFamilyCount\": ${tokenizerPendingFamilies.size}, ")
        append("\"tokenizerMetadataMissingFamilyCount\": ${tokenizerMissingFamilies.size}, ")
        append("\"productionReadyFamilyCount\": ${productionReadyFamilies.size}, ")
        append("\"directSafetensorReadyFamilyCount\": ${directSafetensorReadyFamilies.size}, ")
        append("\"productionReadinessPendingFamilyCount\": ${productionReadinessPendingFamilies.size}, ")
        append("\"directSafetensorPendingFamilyCount\": ${directSafetensorPendingFamilies.size}, ")
        append("\"metadataOnlyFamilyCount\": ${metadataOnlyFamilies.size}, ")
        append("\"selectedFamilyCount\": ${bundledModelFamilies.size}, ")
        append("\"missingProjectCount\": ${missingProjects.size}, ")
        append("\"uncatalogedProjectCount\": ${uncatalogedProjects.size}, ")
        append("\"readinessCounts\": ${jsonIntMap(readinessCounts)}, ")
        append("\"selectedFamilies\": ${jsonStringArray(bundledModelFamilies.sorted())}, ")
        append("\"productionReadyFamilies\": ${jsonStringArray(productionReadyFamilies.sorted())}, ")
        append("\"directSafetensorReadyFamilies\": ${jsonStringArray(directSafetensorReadyFamilies.sorted())}, ")
        append("\"productionReadinessPendingFamilies\": ${jsonStringArray(productionReadinessPendingFamilies.sorted())}, ")
        append("\"directSafetensorPendingFamilies\": ${jsonStringArray(directSafetensorPendingFamilies.sorted())}, ")
        append("\"metadataOnlyFamilies\": ${jsonStringArray(metadataOnlyFamilies.sorted())}, ")
        append("\"missingProjects\": ${jsonStringArray(missingProjects.sorted())}, ")
        append("\"uncatalogedProjects\": ${jsonStringArray(uncatalogedProjects.sorted())}, ")
        append("\"supportOnlyProjects\": ${jsonStringArray(supportOnlyProjects.sorted())}, ")
        append("\"serviceDescriptorIncompleteProjects\": ${jsonStringArray(serviceIncompleteProjects.sorted())}, ")
        append("\"tokenizerMetadataPendingFamilies\": ${jsonStringArray(tokenizerPendingFamilies.sorted())}, ")
        append("\"tokenizerMetadataPendingReasons\": ${jsonStringMap(tokenizerPendingReasons)}, ")
        append("\"tokenizerMetadataMissingFamilies\": ${jsonStringArray(tokenizerMissingFamilies.sorted())}")
        append("}")
    }
}

fun modelFamilyModuleCatalogReportJson(): String {
    val catalogPaths = catalogModelFamilyProjectPaths()
    val includedPaths = includedModelFamilyProjectPaths()
    val missingProjects = modelFamilyModuleCatalogMissingProjects()
    val uncatalogedProjects = modelFamilyModuleCatalogUncatalogedProjects()
    val supportOnlyProjects = modelFamilyModuleCatalogSupportOnlyProjects()
    val serviceIncompleteProjects = modelFamilyModuleCatalogServiceIncompleteProjects()
    val tokenizerMissingFamilies = modelFamilyModuleCatalogTokenizerMetadataMissingFamilies()
    val tokenizerPendingFamilies = modelFamilyModuleCatalogTokenizerMetadataPendingFamilies()
    val tokenizerPendingReasons = modelFamilyModuleCatalogTokenizerMetadataPendingReasons()
    val tokenizerReadyFamilies = modelFamilyModules
        .filter(::modelFamilyModuleTokenizerReady)
        .map { it.id }
        .toSet()
    val productionReadyFamilies = modelFamilyModules
        .filter(::modelFamilyModuleProductionReady)
        .map { it.id }
        .toSet()
    val directSafetensorReadyFamilies = modelFamilyModules
        .filter(::modelFamilyModuleDirectSafetensorReady)
        .map { it.id }
        .toSet()
    val productionReadinessPendingFamilies = modelFamilyModuleCatalogProductionReadinessPendingFamilies()
    val directSafetensorPendingFamilies = modelFamilyModuleCatalogDirectSafetensorPendingFamilies()
    val metadataOnlyFamilies = modelFamilyModules
        .filter(::modelFamilyModuleMetadataOnly)
        .map { it.id }
        .toSet()
    val readinessCounts = modelFamilyModules
        .groupingBy(::modelFamilyModuleReadiness)
        .eachCount()
    val catalogedIncludedCount = catalogPaths.intersect(includedPaths).size
    val passed = missingProjects.isEmpty() &&
            uncatalogedProjects.isEmpty() &&
            serviceIncompleteProjects.isEmpty() &&
            tokenizerMissingFamilies.isEmpty() &&
            productionReadinessPendingFamilies.isEmpty() &&
            directSafetensorPendingFamilies.isEmpty()
    return buildString {
        appendLine("{")
        appendLine("  \"schemaVersion\": 1,")
        appendLine("  \"passed\": $passed,")
        appendLine("  \"catalogFamilyCount\": ${modelFamilyModules.size},")
        appendLine("  \"catalogProjectCount\": ${catalogPaths.size},")
        appendLine("  \"includedModelProjectCount\": ${includedPaths.size},")
        appendLine("  \"includedCatalogedProjectCount\": $catalogedIncludedCount,")
        appendLine("  \"supportOnlyProjectCount\": ${supportOnlyProjects.size},")
        appendLine("  \"serviceDescriptorCompleteProjectCount\": ${catalogedIncludedCount - serviceIncompleteProjects.size},")
        appendLine("  \"serviceDescriptorIncompleteProjectCount\": ${serviceIncompleteProjects.size},")
        appendLine("  \"tokenizerMetadataReadyFamilyCount\": ${tokenizerReadyFamilies.size},")
        appendLine("  \"tokenizerMetadataPendingFamilyCount\": ${tokenizerPendingFamilies.size},")
        appendLine("  \"tokenizerMetadataMissingFamilyCount\": ${tokenizerMissingFamilies.size},")
        appendLine("  \"productionReadyFamilyCount\": ${productionReadyFamilies.size},")
        appendLine("  \"directSafetensorReadyFamilyCount\": ${directSafetensorReadyFamilies.size},")
        appendLine("  \"productionReadinessPendingFamilyCount\": ${productionReadinessPendingFamilies.size},")
        appendLine("  \"directSafetensorPendingFamilyCount\": ${directSafetensorPendingFamilies.size},")
        appendLine("  \"metadataOnlyFamilyCount\": ${metadataOnlyFamilies.size},")
        appendLine("  \"selectedFamilyCount\": ${bundledModelFamilies.size},")
        appendLine("  \"missingProjectCount\": ${missingProjects.size},")
        appendLine("  \"uncatalogedProjectCount\": ${uncatalogedProjects.size},")
        appendLine("  \"readinessCounts\": ${jsonIntMap(readinessCounts)},")
        appendLine("  \"selectedFamilies\": ${jsonStringArray(bundledModelFamilies.sorted())},")
        appendLine("  \"productionReadyFamilies\": ${jsonStringArray(productionReadyFamilies.sorted())},")
        appendLine("  \"directSafetensorReadyFamilies\": ${jsonStringArray(directSafetensorReadyFamilies.sorted())},")
        appendLine("  \"productionReadinessPendingFamilies\": ${jsonStringArray(productionReadinessPendingFamilies.sorted())},")
        appendLine("  \"directSafetensorPendingFamilies\": ${jsonStringArray(directSafetensorPendingFamilies.sorted())},")
        appendLine("  \"metadataOnlyFamilies\": ${jsonStringArray(metadataOnlyFamilies.sorted())},")
        appendLine("  \"missingProjects\": ${jsonStringArray(missingProjects.sorted())},")
        appendLine("  \"uncatalogedProjects\": ${jsonStringArray(uncatalogedProjects.sorted())},")
        appendLine("  \"supportOnlyProjects\": ${jsonStringArray(supportOnlyProjects.sorted())},")
        appendLine("  \"serviceDescriptorIncompleteProjects\": ${jsonStringArray(serviceIncompleteProjects.sorted())},")
        appendLine("  \"tokenizerMetadataPendingFamilies\": ${jsonStringArray(tokenizerPendingFamilies.sorted())},")
        appendLine("  \"tokenizerMetadataPendingReasons\": ${jsonStringMap(tokenizerPendingReasons)},")
        appendLine("  \"tokenizerMetadataMissingFamilies\": ${jsonStringArray(tokenizerMissingFamilies.sorted())},")
        appendLine("  \"catalog\": ${jsonModelFamilyModuleCatalogEntries()},")
        appendLine("  \"includedProjects\": ${jsonIncludedModelFamilyProjectEntries()}")
        appendLine("}")
    }
}

fun modelFamilyBundleReportJson(selected: List<ModelFamilyModule>): String {
    val selectedIds = selected.map { it.id }
    val selectedIdSet = selectedIds.toSet()
    val selectedProfiles = selected.map { it.profile }.distinct().sorted()
    val availableFamilies = modelFamilyModules.map { it.id }.sorted()
    val availableProfiles = modelFamilyModules.map { it.profile }.distinct().sorted()
    val fixtureInventories = modelFamilyFixtureInventories()
    val fixtureInventoryById = fixtureInventories.associateBy { it.module.id }
    val missingRequired = modelFamilyPolicyMissingRequired()
    val selectedForbidden = modelFamilyPolicySelectedForbidden()
    val missingRequiredAliases = modelFamilyPolicyMissingRequiredAliases()
    val selectedForbiddenAliases = modelFamilyPolicySelectedForbiddenAliases()
    val policyViolationCount = modelFamilyPolicyViolationCount(
        missingRequired,
        selectedForbidden,
        missingRequiredAliases,
        selectedForbiddenAliases
    )
    return buildString {
        appendLine("{")
        appendLine("  \"schemaVersion\": ${modelFamilyBundleManifestSchemaVersion.toIntOrNull() ?: 0},")
        appendLine("  \"bundleFingerprint\": ${jsonString(modelFamilyBundleFingerprint(selected))},")
        appendLine(
            "  \"bundlePreset\": ${
                requestedModelFamilyBundlePreset?.id?.let { jsonString(it) } ?: "null"
            },"
        )
        appendLine("  \"availableBundlePresets\": ${jsonStringArray(modelFamilyBundlePresetIds.sorted())},")
        appendLine("  \"activeBundlePresetConformance\": ${jsonModelFamilyBundlePresetConformance()},")
        appendLine("  \"selectorSource\": ${jsonString(requestedModelFamilySelectorSource)},")
        appendLine("  \"explicitSelectors\": ${jsonStringArray(explicitModelFamilyTokens.sorted())},")
        appendLine("  \"presetSelectors\": ${jsonStringArray(presetModelFamilyTokens.sorted())},")
        appendLine("  \"defaultSelectors\": ${jsonStringArray(defaultModelFamilyTokens.sorted())},")
        appendLine("  \"policySource\": ${jsonString(modelFamilyPolicySource)},")
        appendLine("  \"presetRequiredFamilies\": ${jsonStringArray(presetRequiredModelFamilyTokens.sorted())},")
        appendLine("  \"presetForbiddenFamilies\": ${jsonStringArray(presetForbiddenModelFamilyTokens.sorted())},")
        appendLine("  \"presetRequiredAliases\": ${jsonStringArray(presetRequiredModelFamilyAliasTokens.sorted())},")
        appendLine("  \"presetForbiddenAliases\": ${jsonStringArray(presetForbiddenModelFamilyAliasTokens.sorted())},")
        appendLine("  \"explicitRequiredFamilies\": ${jsonStringArray(explicitRequiredModelFamilyTokens.sorted())},")
        appendLine("  \"explicitForbiddenFamilies\": ${jsonStringArray(explicitForbiddenModelFamilyTokens.sorted())},")
        appendLine("  \"explicitRequiredAliases\": ${jsonStringArray(explicitRequiredModelFamilyAliasTokens.sorted())},")
        appendLine("  \"explicitForbiddenAliases\": ${jsonStringArray(explicitForbiddenModelFamilyAliasTokens.sorted())},")
        appendLine("  \"selectors\": ${jsonStringArray(requestedModelFamilyTokens.sorted())},")
        appendLine("  \"requestedFamilies\": ${jsonStringArray(requestedModelFamilyFamilies())},")
        appendLine("  \"requestedProfiles\": ${jsonStringArray(requestedModelFamilyProfiles())},")
        appendLine("  \"requestedAliases\": ${jsonStringArray(requestedModelFamilyAliases())},")
        appendLine("  \"reservedSelectors\": ${jsonStringArray(requestedModelFamilyReservedSelectors())},")
        appendLine("  \"detached\": ${selected.isEmpty()},")
        appendLine("  \"families\": ${jsonStringArray(selectedIds)},")
        appendLine("  \"profiles\": ${jsonStringArray(selectedProfiles)},")
        appendLine("  \"availableFamilies\": ${jsonStringArray(availableFamilies)},")
        appendLine("  \"availableProfiles\": ${jsonStringArray(availableProfiles)},")
        appendLine("  \"availableSelectors\": ${jsonStringArray(availableModelFamilySelectors)},")
        appendLine("  \"moduleCatalog\": ${jsonModelFamilyModuleCatalogSummary()},")
        appendLine("  \"requiredFamilies\": ${jsonStringArray(requiredModelFamilies.sorted())},")
        appendLine("  \"forbiddenFamilies\": ${jsonStringArray(forbiddenModelFamilies.sorted())},")
        appendLine("  \"requiredAliases\": ${jsonStringArray(requiredModelFamilyAliasTokens.sorted())},")
        appendLine("  \"forbiddenAliases\": ${jsonStringArray(forbiddenModelFamilyAliasTokens.sorted())},")
        appendLine("  \"fixtureStatus\": ${jsonModelFamilyFixtureStatus(fixtureInventories)},")
        appendLine("  \"bundlePresets\": ${jsonModelFamilyBundlePresets()},")
        appendLine("  \"bundleAliases\": [")
        val aliasCoverage = modelFamilyBundleAliasCoverage(selectedIdSet)
        aliasCoverage.forEachIndexed { index, coverage ->
            val alias = coverage.alias
            append("    {")
            append("\"id\": ${jsonString(alias.id)}, ")
            append("\"description\": ${jsonString(alias.description)}, ")
            append("\"families\": ${jsonStringArray(alias.familyIds.sorted())}, ")
            append("\"familyCount\": ${alias.familyIds.size}, ")
            append("\"selectedFamilies\": ${jsonStringArray(coverage.selectedFamilyIds.sorted())}, ")
            append("\"selectedCount\": ${coverage.selectedFamilyIds.size}, ")
            append("\"missingFamilies\": ${jsonStringArray(coverage.missingFamilyIds.sorted())}, ")
            append("\"missingCount\": ${coverage.missingFamilyIds.size}, ")
            append("\"complete\": ${coverage.complete}, ")
            append("\"partial\": ${coverage.partial}")
            append("}")
            appendLine(if (index == aliasCoverage.lastIndex) "" else ",")
        }
        appendLine("  ],")
        appendLine("  \"completeAliases\": ${jsonStringArray(aliasCoverage.filter { it.complete }.map { it.alias.id })},")
        appendLine("  \"partialAliases\": ${jsonStringArray(aliasCoverage.filter { it.partial }.map { it.alias.id })},")
        appendLine("  \"policyStatus\": {")
        appendLine("    \"passed\": ${policyViolationCount == 0},")
        appendLine("    \"violationCount\": $policyViolationCount,")
        appendLine("    \"missingRequiredCount\": ${missingRequired.size},")
        appendLine("    \"selectedForbiddenCount\": ${selectedForbidden.size},")
        appendLine("    \"missingRequiredAliasCount\": ${missingRequiredAliases.size},")
        appendLine("    \"selectedForbiddenAliasCount\": ${selectedForbiddenAliases.size},")
        appendLine("    \"missingRequiredAliasFamilyCount\": ${missingRequiredAliases.values.sumOf { it.size }},")
        appendLine("    \"selectedForbiddenAliasFamilyCount\": ${selectedForbiddenAliases.values.sumOf { it.size }}")
        appendLine("  },")
        appendLine("  \"policyViolations\": {")
        appendLine("    \"missingRequired\": ${jsonStringArray(missingRequired.sorted())},")
        appendLine("    \"selectedForbidden\": ${jsonStringArray(selectedForbidden.sorted())},")
        appendLine(
            "    \"missingRequiredAliases\": ${
                jsonAliasPolicyViolations(missingRequiredAliases, "missingFamilies")
            },"
        )
        appendLine(
            "    \"selectedForbiddenAliases\": ${
                jsonAliasPolicyViolations(selectedForbiddenAliases, "selectedFamilies")
            }"
        )
        appendLine("  },")
        appendLine("  \"selection\": [")
        modelFamilyModules.sortedBy { it.id }.forEachIndexed { index, module ->
            append("    {")
            append("\"id\": ${jsonString(module.id)}, ")
            append("\"profile\": ${jsonString(module.profile)}, ")
            append("\"path\": ${jsonString(module.path)}, ")
            append("\"selected\": ${module.id in bundledModelFamilies}, ")
            append("\"fixture\": ${jsonModelFamilyFixtureSelection(fixtureInventoryById.getValue(module.id))}")
            append("}")
            appendLine(if (index == modelFamilyModules.lastIndex) "" else ",")
        }
        appendLine("  ]")
        appendLine("}")
    }
}

val generatedModelFamilyBundleDir = layout.buildDirectory.dir("generated/resources/model-family-bundle")
val modelFamilyBundleManifestFile = generatedModelFamilyBundleDir.map {
    it.file("META-INF/gollek-model-family-bundle.properties")
}
val modelFamilyBundleReportFile = layout.buildDirectory.file("reports/gollek/model-family-bundle.json")
val modelFamilyModuleCatalogReportFile =
    layout.buildDirectory.file("reports/gollek/model-family-module-catalog.json")
val modelFamilyFixtureReportFile = layout.buildDirectory.file("reports/gollek/model-family-fixtures.json")
val modelFamilyFixtureFingerprintReportFile =
    layout.buildDirectory.file("reports/gollek/model-family-fixture-fingerprints.json")
val modelFamilyBundleLockDriftReportFile =
    layout.buildDirectory.file("reports/gollek/model-family-bundle-lock-drift.json")
val modelFamilyFixtureFingerprintLockDriftReportFile =
    layout.buildDirectory.file("reports/gollek/model-family-fixture-fingerprint-lock-drift.json")

fun modelFamilyFixtureRootDirs() = modelFamilyModules.map { module ->
    modelFamilyProjectDir(module).resolve("src/test/resources/model-family-fixtures")
}

val generateModelFamilyBundleManifest = tasks.register("generateModelFamilyBundleManifest") {
    group = "build"
    description = "Generate packaged model-family bundle metadata for CLI runtime diagnostics."

    val outputFile = modelFamilyBundleManifestFile
    outputs.file(outputFile)
    inputs.property("requestedModelFamilyTokens", requestedModelFamilyTokens.sorted().joinToString(","))
    inputs.property("requestedModelFamilySelectorSource", requestedModelFamilySelectorSource)
    inputs.property("explicitModelFamilyTokens", explicitModelFamilyTokens.sorted().joinToString(","))
    inputs.property("presetModelFamilyTokens", presetModelFamilyTokens.sorted().joinToString(","))
    inputs.property("defaultModelFamilyTokens", defaultModelFamilyTokens.sorted().joinToString(","))
    inputs.property("modelFamilyPolicySource", modelFamilyPolicySource)
    inputs.property("presetRequiredModelFamilyTokens", presetRequiredModelFamilyTokens.sorted().joinToString(","))
    inputs.property("presetForbiddenModelFamilyTokens", presetForbiddenModelFamilyTokens.sorted().joinToString(","))
    inputs.property("presetRequiredModelFamilyAliasTokens", presetRequiredModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("presetForbiddenModelFamilyAliasTokens", presetForbiddenModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("explicitRequiredModelFamilyTokens", explicitRequiredModelFamilyTokens.sorted().joinToString(","))
    inputs.property("explicitForbiddenModelFamilyTokens", explicitForbiddenModelFamilyTokens.sorted().joinToString(","))
    inputs.property("explicitRequiredModelFamilyAliasTokens", explicitRequiredModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("explicitForbiddenModelFamilyAliasTokens", explicitForbiddenModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("requestedModelFamilyBundlePreset", requestedModelFamilyBundlePreset?.id ?: "")
    inputs.property("modelFamilyBundlePresets", modelFamilyBundlePresetInput())
    inputs.property("bundledModelFamilies", bundledModelFamilies.sorted().joinToString(","))
    inputs.property("requiredModelFamilies", requiredModelFamilies.sorted().joinToString(","))
    inputs.property("forbiddenModelFamilies", forbiddenModelFamilies.sorted().joinToString(","))
    inputs.property("requiredModelFamilyAliases", requiredModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("forbiddenModelFamilyAliases", forbiddenModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property(
        "productionReadinessPendingFamilies",
        modelFamilyModuleCatalogProductionReadinessPendingFamilies().sorted().joinToString(",")
    )
    inputs.property(
        "directSafetensorPendingFamilies",
        modelFamilyModuleCatalogDirectSafetensorPendingFamilies().sorted().joinToString(",")
    )
    inputs.property("pendingModelFamilyTokenizerMetadata", pendingModelFamilyTokenizerMetadataInput())
    inputs.property("manifestSchemaVersion", modelFamilyBundleManifestSchemaVersion)
    inputs.files(modelFamilyFixtureRootDirs())
    inputs.property("requiredModelFamilyFixtureTokens", requiredModelFamilyFixtureTokens.sorted().joinToString(","))
    inputs.property("requiredModelFamilyFixtures", requiredModelFamilyFixtures.sorted().joinToString(","))
    inputs.property(
        "modelFamilyProfiles",
        modelFamilyModules.joinToString(",") { "${it.id}:${it.profile}:${it.path}" }
    )
    inputs.property(
        "modelFamilyBundleAliases",
        modelFamilyBundleAliases.joinToString(",") { "${it.id}:${it.familyIds.sorted().joinToString("+")}" }
    )
    inputs.property("pendingModelFamilyTokenizerMetadata", pendingModelFamilyTokenizerMetadataInput())

    doLast {
        val selected = selectedModelFamilyModules()
        val missingRequired = modelFamilyPolicyMissingRequired()
        val selectedForbidden = modelFamilyPolicySelectedForbidden()
        val missingRequiredAliases = modelFamilyPolicyMissingRequiredAliases()
        val selectedForbiddenAliases = modelFamilyPolicySelectedForbiddenAliases()
        val policyViolationCount = modelFamilyPolicyViolationCount(
            missingRequired,
            selectedForbidden,
            missingRequiredAliases,
            selectedForbiddenAliases
        )
        val fixtureSnapshot = modelFamilyFixtureLockSnapshot()
        val productionReadinessPendingFamilies = modelFamilyModuleCatalogProductionReadinessPendingFamilies()
        val directSafetensorPendingFamilies = modelFamilyModuleCatalogDirectSafetensorPendingFamilies()
        val file = outputFile.get().asFile
        file.parentFile.mkdirs()
        file.writeText(
            buildString {
                appendLine("# Generated by :ui:gollek-cli:generateModelFamilyBundleManifest")
                appendLine("schemaVersion=$modelFamilyBundleManifestSchemaVersion")
                appendLine("bundleFingerprint=${modelFamilyBundleFingerprint(selected)}")
                appendLine("bundlePreset=${requestedModelFamilyBundlePreset?.id ?: ""}")
                appendLine("selectorSource=$requestedModelFamilySelectorSource")
                appendLine("explicitSelectors=${explicitModelFamilyTokens.sorted().joinToString(",")}")
                appendLine("presetSelectors=${presetModelFamilyTokens.sorted().joinToString(",")}")
                appendLine("defaultSelectors=${defaultModelFamilyTokens.sorted().joinToString(",")}")
                appendLine("policySource=$modelFamilyPolicySource")
                appendLine("presetRequiredFamilies=${presetRequiredModelFamilyTokens.sorted().joinToString(",")}")
                appendLine("presetForbiddenFamilies=${presetForbiddenModelFamilyTokens.sorted().joinToString(",")}")
                appendLine("presetRequiredAliases=${presetRequiredModelFamilyAliasTokens.sorted().joinToString(",")}")
                appendLine("presetForbiddenAliases=${presetForbiddenModelFamilyAliasTokens.sorted().joinToString(",")}")
                appendLine("explicitRequiredFamilies=${explicitRequiredModelFamilyTokens.sorted().joinToString(",")}")
                appendLine("explicitForbiddenFamilies=${explicitForbiddenModelFamilyTokens.sorted().joinToString(",")}")
                appendLine("explicitRequiredAliases=${explicitRequiredModelFamilyAliasTokens.sorted().joinToString(",")}")
                appendLine("explicitForbiddenAliases=${explicitForbiddenModelFamilyAliasTokens.sorted().joinToString(",")}")
                appendLine("requiresDirectSafetensorRuntime=$requiresDirectSafetensorRuntime")
                appendLine("productionReadinessPassed=${productionReadinessPendingFamilies.isEmpty()}")
                appendLine("productionReadinessPendingCount=${productionReadinessPendingFamilies.size}")
                appendLine("productionReadinessPendingFamilies=${productionReadinessPendingFamilies.sorted().joinToString(",")}")
                appendLine("directSafetensorReadinessPassed=${directSafetensorPendingFamilies.isEmpty()}")
                appendLine("directSafetensorPendingCount=${directSafetensorPendingFamilies.size}")
                appendLine("directSafetensorPendingFamilies=${directSafetensorPendingFamilies.sorted().joinToString(",")}")
                appendLine("selectors=${requestedModelFamilyTokens.sorted().joinToString(",")}")
                appendLine("requiredFamilies=${requiredModelFamilies.sorted().joinToString(",")}")
                appendLine("forbiddenFamilies=${forbiddenModelFamilies.sorted().joinToString(",")}")
                appendLine("requiredAliases=${requiredModelFamilyAliasTokens.sorted().joinToString(",")}")
                appendLine("forbiddenAliases=${forbiddenModelFamilyAliasTokens.sorted().joinToString(",")}")
                appendLine("policyPassed=${policyViolationCount == 0}")
                appendLine("policyViolationCount=$policyViolationCount")
                appendLine("missingRequiredFamilies=${missingRequired.sorted().joinToString(",")}")
                appendLine("selectedForbiddenFamilies=${selectedForbidden.sorted().joinToString(",")}")
                appendLine("missingRequiredAliases=${missingRequiredAliases.keys.sorted().joinToString(",")}")
                appendLine("selectedForbiddenAliases=${selectedForbiddenAliases.keys.sorted().joinToString(",")}")
                appendLine("fixtureRequiredSelectors=${fixtureSnapshot.requiredSelectors.joinToString(",")}")
                appendLine("fixtureRequiredFamilies=${fixtureSnapshot.requiredFamilies.joinToString(",")}")
                appendLine("fixturePassed=${fixtureSnapshot.passed}")
                appendLine("fixtureRequiredFingerprint=${fixtureSnapshot.requiredFingerprint}")
                appendLine("fixtureInventoryFingerprint=${fixtureSnapshot.inventoryFingerprint}")
                appendLine("fixtureAvailableFamilyCount=${fixtureSnapshot.availableFamilyCount}")
                appendLine("fixtureFamilyCount=${fixtureSnapshot.fixtureFamilyCount}")
                appendLine("fixtureRequiredFamilyCount=${fixtureSnapshot.requiredFamilyCount}")
                appendLine("fixtureRequiredPassedCount=${fixtureSnapshot.requiredPassedCount}")
                appendLine("fixtureMissingRequiredCount=${fixtureSnapshot.missingRequiredCount}")
                appendLine("fixtureProblemFamilyCount=${fixtureSnapshot.problemFamilyCount}")
                appendLine("fixtureMissingRequiredFamilies=${fixtureSnapshot.missingRequiredFamilies.joinToString(",")}")
                appendLine("fixtureProblemFamilies=${fixtureSnapshot.problemFamilies.joinToString(",")}")
                for ((aliasId, missing) in missingRequiredAliases.toSortedMap()) {
                    appendLine("missingRequiredAlias.${aliasId}.families=${missing.sorted().joinToString(",")}")
                }
                for ((aliasId, selectedFamilies) in selectedForbiddenAliases.toSortedMap()) {
                    appendLine(
                        "selectedForbiddenAlias.${aliasId}.families=${
                            selectedFamilies.sorted().joinToString(",")
                        }"
                    )
                }
                appendLine("detached=${selected.isEmpty()}")
                appendLine("familyCount=${selected.size}")
                appendLine("families=${selected.joinToString(",") { it.id }}")
                appendLine("profiles=${selected.map { it.profile }.distinct().sorted().joinToString(",")}")
                appendLine("availableFamilies=${modelFamilyModules.map { it.id }.sorted().joinToString(",")}")
                appendLine("availableProfiles=${modelFamilyModules.map { it.profile }.distinct().sorted().joinToString(",")}")
                appendLine("availableSelectors=${availableModelFamilySelectors.joinToString(",")}")
                appendLine("tokenizerMetadataPendingFamilies=${pendingModelFamilyTokenizerMetadata.sorted().joinToString(",")}")
                appendLine("availableBundlePresets=${modelFamilyBundlePresetIds.sorted().joinToString(",")}")
                appendLine("bundlePresets=${modelFamilyBundlePresets.map { it.id }.sorted().joinToString(",")}")
                for (preset in modelFamilyBundlePresets.sortedBy { it.id }) {
                    val validation = modelFamilyBundlePresetValidation(preset)
                    appendLine("bundlePreset.${preset.id}.description=${preset.description}")
                    appendLine("bundlePreset.${preset.id}.selectors=${preset.selectors.sorted().joinToString(",")}")
                    appendLine(
                        "bundlePreset.${preset.id}.requiredFamilies=${
                            preset.requiredFamilies.sorted().joinToString(",")
                        }"
                    )
                    appendLine(
                        "bundlePreset.${preset.id}.forbiddenFamilies=${
                            preset.forbiddenFamilies.sorted().joinToString(",")
                        }"
                    )
                    appendLine(
                        "bundlePreset.${preset.id}.requiredAliases=${
                            preset.requiredAliases.sorted().joinToString(",")
                        }"
                    )
                    appendLine(
                        "bundlePreset.${preset.id}.forbiddenAliases=${
                            preset.forbiddenAliases.sorted().joinToString(",")
                        }"
                    )
                    appendLine(
                        "bundlePreset.${preset.id}.selectedFamilies=${
                            validation.selectedFamilyIds.sorted().joinToString(",")
                        }"
                    )
                    appendLine("bundlePreset.${preset.id}.selectedCount=${validation.selectedFamilyIds.size}")
                    appendLine(
                        "bundlePreset.${preset.id}.productionTokenizerMetadataRequired=${
                            validation.productionTokenizerMetadataRequired
                        }"
                    )
                    appendLine(
                        "bundlePreset.${preset.id}.productionTokenizerMetadataReady=${
                            validation.productionTokenizerMetadataReady
                        }"
                    )
                    appendLine(
                        "bundlePreset.${preset.id}.productionSafetyPassed=${validation.productionSafetyPassed}"
                    )
                    appendLine(
                        "bundlePreset.${preset.id}.productionSafetyViolationCount=${
                            validation.productionSafetyViolationCount
                        }"
                    )
                    appendLine(
                        "bundlePreset.${preset.id}.pendingTokenizerFamilies=${
                            validation.pendingTokenizerFamilyIds.sorted().joinToString(",")
                        }"
                    )
                    for ((familyId, reason) in validation.pendingTokenizerReasons.toSortedMap()) {
                        appendLine("bundlePreset.${preset.id}.pendingTokenizerFamily.${familyId}.reason=$reason")
                    }
                    appendLine("bundlePreset.${preset.id}.policyPassed=${validation.passed}")
                    appendLine("bundlePreset.${preset.id}.policyViolationCount=${validation.violationCount}")
                    appendLine(
                        "bundlePreset.${preset.id}.missingRequiredFamilies=${
                            validation.missingRequiredFamilyIds.sorted().joinToString(",")
                        }"
                    )
                    appendLine("bundlePreset.${preset.id}.missingRequiredCount=${validation.missingRequiredFamilyIds.size}")
                    appendLine(
                        "bundlePreset.${preset.id}.selectedForbiddenFamilies=${
                            validation.selectedForbiddenFamilyIds.sorted().joinToString(",")
                        }"
                    )
                    appendLine("bundlePreset.${preset.id}.selectedForbiddenCount=${validation.selectedForbiddenFamilyIds.size}")
                    appendLine(
                        "bundlePreset.${preset.id}.missingRequiredAliases=${
                            validation.missingRequiredAliases.keys.sorted().joinToString(",")
                        }"
                    )
                    appendLine("bundlePreset.${preset.id}.missingRequiredAliasCount=${validation.missingRequiredAliases.size}")
                    appendLine(
                        "bundlePreset.${preset.id}.selectedForbiddenAliases=${
                            validation.selectedForbiddenAliases.keys.sorted().joinToString(",")
                        }"
                    )
                    appendLine("bundlePreset.${preset.id}.selectedForbiddenAliasCount=${validation.selectedForbiddenAliases.size}")
                    for ((aliasId, missing) in validation.missingRequiredAliases.toSortedMap()) {
                        appendLine(
                            "bundlePreset.${preset.id}.missingRequiredAlias.${aliasId}.families=${
                                missing.sorted().joinToString(",")
                            }"
                        )
                    }
                    for ((aliasId, selectedFamilies) in validation.selectedForbiddenAliases.toSortedMap()) {
                        appendLine(
                            "bundlePreset.${preset.id}.selectedForbiddenAlias.${aliasId}.families=${
                                selectedFamilies.sorted().joinToString(",")
                            }"
                        )
                    }
                }
                appendLine("bundleAliases=${modelFamilyBundleAliases.map { it.id }.sorted().joinToString(",")}")
                for (alias in modelFamilyBundleAliases.sortedBy { it.id }) {
                    appendLine("bundleAlias.${alias.id}.description=${alias.description}")
                    appendLine("bundleAlias.${alias.id}.families=${alias.familyIds.sorted().joinToString(",")}")
                    appendLine("bundleAlias.${alias.id}.familyCount=${alias.familyIds.size}")
                }
                for (module in modelFamilyModules.sortedBy { it.id }) {
                    appendLine("family.${module.id}.selected=${module.id in bundledModelFamilies}")
                    appendLine("family.${module.id}.profile=${module.profile}")
                    appendLine("family.${module.id}.path=${module.path}")
                    val pendingReason = modelFamilyTokenizerMetadataPendingReason(module.id)
                    if (pendingReason.isNotBlank()) {
                        appendLine("family.${module.id}.tokenizerMetadataPendingReason=$pendingReason")
                    }
                }
            },
            Charsets.UTF_8
        )
    }
}

dependencies {
    fun includeModelFamilies() {
        for (module in modelFamilyModules) {
            if (module.id in bundledModelFamilies) {
                val moduleProject = modelFamilyProject(module)
                    ?: throw GradleException(
                        "Selected model-family project ${module.path} for ${module.id} is not included"
                    )
                implementation(moduleProject)
            }
        }
    }

    implementation(platform("io.quarkus.platform:quarkus-bom:$quarkusVersion"))

    implementation("io.quarkus:quarkus-arc")
    implementation("io.quarkus:quarkus-cache")
    implementation("io.quarkus:quarkus-jackson")
    implementation("io.quarkus:quarkus-logging-json")
    implementation("io.quarkus:quarkus-picocli")
    implementation("io.quarkus:quarkus-mutiny")
    implementation("io.quarkus:quarkus-smallrye-health")

    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310")
    implementation("org.apache.commons:commons-lang3:3.14.0")
    implementation("commons-io:commons-io:2.18.0")
    implementation("org.jline:jline-console:3.26.3")
    implementation("org.jline:jline-reader:3.26.3")
    implementation("org.jline:jline-terminal:3.26.3")
    implementation("org.jline:jline-terminal-jna:3.26.3")
    implementation("com.github.albfernandez:juniversalchardet:2.4.0")
    implementation("com.google.ai.edge.litertlm:litertlm-jvm:0.11.0")

    implementation(project(":sdk:gollek-sdk"))
    implementation(project(":sdk:gollek-sdk-agent"))
    implementation(project(":sdk:gollek-sdk-api"))
    implementation(project(":sdk:gollek-sdk-core"))
    if (findProject(":sdk:gollek-sdk-session") != null) {
        implementation(project(":sdk:gollek-sdk-session"))
    }
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":spi:gollek-spi-provider"))
    implementation(project(":spi:gollek-spi-runtime"))
    implementation(project(":core:gollek-model-repository"))
    implementation(project(":core:gollek-runtime-config"))
    implementation(project(":core:gollek-provider-routing"))
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(project(":core:plugin:gollek-plugin-kernel-core"))
    implementation(project(":core:plugin:gollek-plugin-runner-core"))
    implementation(project(":core:plugin:gollek-plugin-core"))
    implementation(project(":core:plugin:gollek-plugin-runner-gguf"))
    implementation(project(":core:gollek-model-repo-hf"))
    implementation(project(":models:gollek-model-gemma4"))
    includeModelFamilies()
    implementation(project(":core:gollek-model-repo-kaggle"))
    implementation(project(":core:gollek-model-repo-local"))
    implementation(project(":plugins:gollek-plugin-mcp"))
    implementation(project(":plugins:log-parser"))
    implementation(project(":runner:gguf:gollek-gguf-core"))
    implementation(project(":runner:litert:gollek-runner-litert"))
    implementation(project(":runner:onnx:gollek-runner-onnx"))
    if (findProject(":suling") != null) {
        implementation(project(":suling"))
    }
    implementation(project(":runner:safetensor:gollek-safetensor-engine"))
    implementation(project(":runner:safetensor:gollek-safetensor-loader"))
    implementation(project(":runner:safetensor:gollek-safetensor-spi"))
    implementation(project(":backend:metal:gollek-backend-metal"))
    implementation(project(":ml:gollek-ml-api"))

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("io.quarkus:quarkus-junit5")
    testImplementation("io.quarkus:quarkus-junit5-mockito")
}

sourceSets {
    main {
        java {
            setSrcDirs(listOf("src/main/java"))
            exclude("tech/kayys/gollek/cli/NewGollekCLI.java")
        }
        resources {
            setSrcDirs(listOf("src/main/resources"))
            srcDir(generatedModelFamilyBundleDir)
        }
    }
}

tasks.processResources {
    dependsOn(generateModelFamilyBundleManifest)
    filesMatching("META-INF/gollek-version.properties") {
        filter { line: String ->
            line.replace("\${project.version}", project.version.toString())
        }
    }
}

tasks.jar {
    archiveBaseName.set("gollek")
}

tasks.register("printModelFamilyBundle") {
    group = "help"
    description = "Print selected model-family modules for this CLI build."
    doLast {
        val selected = selectedModelFamilyModules()
        val missingRequired = modelFamilyPolicyMissingRequired()
        val selectedForbidden = modelFamilyPolicySelectedForbidden()
        val missingRequiredAliases = modelFamilyPolicyMissingRequiredAliases()
        val selectedForbiddenAliases = modelFamilyPolicySelectedForbiddenAliases()
        val policyViolationCount = modelFamilyPolicyViolationCount(
            missingRequired,
            selectedForbidden,
            missingRequiredAliases,
            selectedForbiddenAliases
        )
        val conformance = modelFamilyBundlePresetConformanceValidation()
        println("Requested model-family selectors: ${requestedModelFamilyTokens.sorted().joinToString(", ")}")
        println(
            "Model-family bundle preset: ${
                requestedModelFamilyBundlePreset?.let { "${it.id} - ${it.description}" } ?: "none"
            }"
        )
        println("Preset conformance: ${conformance.compactSummary}")
        if (conformance.hasPreset && !conformance.cleanPresetBuild) {
            println("Preset conformance details:")
            if (conformance.explicitSelectorOverride) {
                println("  - explicit selector override: yes")
            }
            if (conformance.explicitPolicyOverride) {
                println("  - explicit policy override: yes")
            }
            if (conformance.selectorAdditions.isNotEmpty()) {
                println("  - selector additions: ${conformance.selectorAdditions.sorted().joinToString(", ")}")
            }
            if (conformance.selectorOmissions.isNotEmpty()) {
                println("  - selector omissions: ${conformance.selectorOmissions.sorted().joinToString(", ")}")
            }
            if (conformance.requiredFamilyAdditions.isNotEmpty()) {
                println("  - required family additions: ${conformance.requiredFamilyAdditions.sorted().joinToString(", ")}")
            }
            if (conformance.requiredFamilyOmissions.isNotEmpty()) {
                println("  - required family omissions: ${conformance.requiredFamilyOmissions.sorted().joinToString(", ")}")
            }
            if (conformance.forbiddenFamilyAdditions.isNotEmpty()) {
                println("  - forbidden family additions: ${conformance.forbiddenFamilyAdditions.sorted().joinToString(", ")}")
            }
            if (conformance.forbiddenFamilyOmissions.isNotEmpty()) {
                println("  - forbidden family omissions: ${conformance.forbiddenFamilyOmissions.sorted().joinToString(", ")}")
            }
            if (conformance.requiredAliasAdditions.isNotEmpty()) {
                println("  - required alias additions: ${conformance.requiredAliasAdditions.sorted().joinToString(", ")}")
            }
            if (conformance.requiredAliasOmissions.isNotEmpty()) {
                println("  - required alias omissions: ${conformance.requiredAliasOmissions.sorted().joinToString(", ")}")
            }
            if (conformance.forbiddenAliasAdditions.isNotEmpty()) {
                println("  - forbidden alias additions: ${conformance.forbiddenAliasAdditions.sorted().joinToString(", ")}")
            }
            if (conformance.forbiddenAliasOmissions.isNotEmpty()) {
                println("  - forbidden alias omissions: ${conformance.forbiddenAliasOmissions.sorted().joinToString(", ")}")
            }
        }
        println("Available bundle presets: ${modelFamilyBundlePresetIds.sorted().joinToString(", ")}")
        println("Requested selector families: ${requestedModelFamilyFamilies().joinToString(", ").ifBlank { "none" }}")
        println("Requested selector profiles: ${requestedModelFamilyProfiles().joinToString(", ").ifBlank { "none" }}")
        println("Requested selector aliases: ${requestedModelFamilyAliases().joinToString(", ").ifBlank { "none" }}")
        println("Model-family bundle fingerprint: ${modelFamilyBundleFingerprint(selected)}")
        println("Bundle profiles: ${selected.map { it.profile }.distinct().sorted().joinToString(", ").ifBlank { "none" }}")
        println("Bundle selector aliases: ${modelFamilyBundleAliasIds.sorted().joinToString(", ")}")
        println("Complete selector aliases: ${joinedModelFamilyBundleAliasCoverage(bundledModelFamilies) { it.complete }}")
        println("Partial selector aliases: ${joinedModelFamilyBundleAliasCoverage(bundledModelFamilies) { it.partial }}")
        println("Policy source: $modelFamilyPolicySource")
        println("Policy required families: ${requiredModelFamilies.sorted().joinToString(", ").ifBlank { "none" }}")
        println("Policy forbidden families: ${forbiddenModelFamilies.sorted().joinToString(", ").ifBlank { "none" }}")
        println(
            "Policy required aliases: ${
                requiredModelFamilyAliasTokens.sorted().joinToString(", ").ifBlank { "none" }
            }"
        )
        println(
            "Policy forbidden aliases: ${
                forbiddenModelFamilyAliasTokens.sorted().joinToString(", ").ifBlank { "none" }
            }"
        )
        println(
            "Policy status: ${
                if (policyViolationCount == 0) "passed" else "failed ($policyViolationCount violation(s))"
            }"
        )
        if (policyViolationCount > 0) {
            println("Policy violations:")
            if (missingRequired.isNotEmpty()) {
                println("  - missing required: ${missingRequired.sorted().joinToString(", ")}")
            }
            if (selectedForbidden.isNotEmpty()) {
                println("  - selected forbidden: ${selectedForbidden.sorted().joinToString(", ")}")
            }
            for ((aliasId, missing) in missingRequiredAliases) {
                println("  - missing required alias $aliasId: ${missing.sorted().joinToString(", ")}")
            }
            for ((aliasId, selectedFamilies) in selectedForbiddenAliases) {
                println("  - selected forbidden alias $aliasId: ${selectedFamilies.sorted().joinToString(", ")}")
            }
        }
        println("Bundled model families (${selected.size}):")
        if (selected.isEmpty()) {
            println("  - none (model-family plugins detached)")
        } else {
            selected.forEach { module ->
                println("  - ${module.id} [${module.profile}] ${module.path}")
            }
        }
    }
}

tasks.register("printModelFamilyBundlePresets") {
    group = "help"
    description = "Print production model-family bundle presets and their simulated policy status."
    doLast {
        val validations = modelFamilyBundlePresetValidations()
        println("Model-family bundle presets (${validations.size}):")
        for (validation in validations) {
            val preset = validation.preset
            val presetStatus = when {
                validation.passed && validation.productionSafetyPassed -> "passed"
                !validation.passed && !validation.productionSafetyPassed ->
                    "failed (${validation.violationCount} policy violation(s), " +
                            "${validation.productionSafetyViolationCount} production safety violation(s))"
                !validation.passed -> "failed (${validation.violationCount} policy violation(s))"
                else -> "failed (${validation.productionSafetyViolationCount} production safety violation(s))"
            }
            println(
                "  - ${preset.id}: $presetStatus"
            )
            println("    description: ${preset.description}")
            println("    selectors: ${preset.selectors.sorted().joinToString(", ").ifBlank { "none" }}")
            println("    selected families (${validation.selectedFamilyIds.size}): ${compactFamilyList(validation.selectedFamilyIds)}")
            println(
                "    production tokenizer metadata: ${
                    if (validation.productionTokenizerMetadataRequired) {
                        if (validation.productionTokenizerMetadataReady) {
                            "ready"
                        } else {
                            "blocked (${validation.pendingTokenizerFamilyIds.sorted().joinToString(", ")})"
                        }
                    } else {
                        "not required"
                    }
                }"
            )
            println("    required families: ${preset.requiredFamilies.sorted().joinToString(", ").ifBlank { "none" }}")
            println("    forbidden families: ${preset.forbiddenFamilies.sorted().joinToString(", ").ifBlank { "none" }}")
            println("    required aliases: ${preset.requiredAliases.sorted().joinToString(", ").ifBlank { "none" }}")
            println("    forbidden aliases: ${preset.forbiddenAliases.sorted().joinToString(", ").ifBlank { "none" }}")

            if (!validation.passed) {
                validation.configurationProblems.forEach { problem ->
                    println("    violation: $problem")
                }
                if (validation.missingRequiredFamilyIds.isNotEmpty()) {
                    println("    missing required: ${validation.missingRequiredFamilyIds.sorted().joinToString(", ")}")
                }
                if (validation.selectedForbiddenFamilyIds.isNotEmpty()) {
                    println("    selected forbidden: ${validation.selectedForbiddenFamilyIds.sorted().joinToString(", ")}")
                }
                for ((aliasId, missing) in validation.missingRequiredAliases) {
                    println("    missing required alias $aliasId: ${missing.sorted().joinToString(", ")}")
                }
                for ((aliasId, selectedFamilies) in validation.selectedForbiddenAliases) {
                    println("    selected forbidden alias $aliasId: ${selectedFamilies.sorted().joinToString(", ")}")
                }
            }
            if (!validation.productionSafetyPassed) {
                for ((familyId, reason) in validation.pendingTokenizerReasons.toSortedMap()) {
                    println("    production tokenizer pending $familyId: $reason")
                }
            }
        }
    }
}

tasks.register("printModelFamilyBundleJson") {
    group = "help"
    description = "Print selected model-family modules as JSON for CI/release automation."
    dependsOn("writeModelFamilyBundleReport")
    doLast {
        println(modelFamilyBundleReportFile.get().asFile.readText(Charsets.UTF_8))
    }
}

tasks.register("printModelFamilyModuleCatalogJson") {
    group = "help"
    description = "Print attachable model-family module catalog JSON for CI/release automation."
    dependsOn("writeModelFamilyModuleCatalogReport")
    doLast {
        println(modelFamilyModuleCatalogReportFile.get().asFile.readText(Charsets.UTF_8))
    }
}

tasks.register("writeModelFamilyModuleCatalogReport") {
    group = "help"
    description = "Write attachable model-family module catalog JSON for CI/release automation."

    val outputFile = modelFamilyModuleCatalogReportFile
    outputs.file(outputFile)
    inputs.property("catalogModelFamilyProjectPaths", catalogModelFamilyProjectPaths().sorted().joinToString(","))
    inputs.property("includedModelFamilyProjectPaths", includedModelFamilyProjectPaths().sorted().joinToString(","))
    inputs.property("pendingModelFamilyTokenizerMetadata", pendingModelFamilyTokenizerMetadataInput())
    inputs.property(
        "includedCatalogedModelFamilyProjectPaths",
        includedCatalogedModelFamilyProjectPaths().sorted().joinToString(",")
    )
    inputs.property("bundledModelFamilies", bundledModelFamilies.sorted().joinToString(","))
    inputs.files(includedModelFamilyProjectPaths().flatMap { path ->
        listOf(
            modelFamilyProjectResourceFile(path, modelFamilyPluginDescriptorRelativePath),
            modelFamilyProjectResourceFile(path, modelFamilyServiceDescriptorRelativePath),
            modelFamilyProjectResourceFile(path, gollekPluginServiceDescriptorRelativePath)
        )
    })

    doLast {
        val file = outputFile.get().asFile
        file.parentFile.mkdirs()
        file.writeText(modelFamilyModuleCatalogReportJson(), Charsets.UTF_8)
    }
}

tasks.register("writeModelFamilyBundleReport") {
    group = "help"
    description = "Write selected model-family modules as JSON for CI/release automation."

    val outputFile = modelFamilyBundleReportFile
    val fixtureRoots = modelFamilyModules.map { module ->
        modelFamilyProjectDir(module).resolve("src/test/resources/model-family-fixtures")
    }
    outputs.file(outputFile)
    inputs.files(fixtureRoots)
    inputs.property("requestedModelFamilyBundlePreset", requestedModelFamilyBundlePreset?.id ?: "")
    inputs.property("modelFamilyBundlePresets", modelFamilyBundlePresetInput())
    inputs.property("requestedModelFamilyTokens", requestedModelFamilyTokens.sorted().joinToString(","))
    inputs.property("requiredModelFamilyFixtureTokens", requiredModelFamilyFixtureTokens.sorted().joinToString(","))
    inputs.property("requiredModelFamilyFixtures", requiredModelFamilyFixtures.sorted().joinToString(","))
    inputs.property("requestedModelFamilySelectorSource", requestedModelFamilySelectorSource)
    inputs.property("explicitModelFamilyTokens", explicitModelFamilyTokens.sorted().joinToString(","))
    inputs.property("presetModelFamilyTokens", presetModelFamilyTokens.sorted().joinToString(","))
    inputs.property("defaultModelFamilyTokens", defaultModelFamilyTokens.sorted().joinToString(","))
    inputs.property("modelFamilyPolicySource", modelFamilyPolicySource)
    inputs.property("presetRequiredModelFamilyTokens", presetRequiredModelFamilyTokens.sorted().joinToString(","))
    inputs.property("presetForbiddenModelFamilyTokens", presetForbiddenModelFamilyTokens.sorted().joinToString(","))
    inputs.property("presetRequiredModelFamilyAliasTokens", presetRequiredModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("presetForbiddenModelFamilyAliasTokens", presetForbiddenModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("explicitRequiredModelFamilyTokens", explicitRequiredModelFamilyTokens.sorted().joinToString(","))
    inputs.property("explicitForbiddenModelFamilyTokens", explicitForbiddenModelFamilyTokens.sorted().joinToString(","))
    inputs.property("explicitRequiredModelFamilyAliasTokens", explicitRequiredModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("explicitForbiddenModelFamilyAliasTokens", explicitForbiddenModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("bundledModelFamilies", bundledModelFamilies.sorted().joinToString(","))
    inputs.property("requiredModelFamilies", requiredModelFamilies.sorted().joinToString(","))
    inputs.property("forbiddenModelFamilies", forbiddenModelFamilies.sorted().joinToString(","))
    inputs.property("requiredModelFamilyAliases", requiredModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("forbiddenModelFamilyAliases", forbiddenModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property(
        "missingRequiredModelFamilyAliases",
        modelFamilyPolicyMissingRequiredAliases().map { (aliasId, families) ->
            "$aliasId:${families.sorted().joinToString("+")}"
        }.joinToString(",")
    )
    inputs.property(
        "selectedForbiddenModelFamilyAliases",
        modelFamilyPolicySelectedForbiddenAliases().map { (aliasId, families) ->
            "$aliasId:${families.sorted().joinToString("+")}"
        }.joinToString(",")
    )
    inputs.property("manifestSchemaVersion", modelFamilyBundleManifestSchemaVersion)
    inputs.property(
        "modelFamilyProfiles",
        modelFamilyModules.joinToString(",") { "${it.id}:${it.profile}:${it.path}" }
    )
    inputs.property(
        "modelFamilyBundleAliases",
        modelFamilyBundleAliases.joinToString(",") { "${it.id}:${it.familyIds.sorted().joinToString("+")}" }
    )

    doLast {
        val file = outputFile.get().asFile
        file.parentFile.mkdirs()
        file.writeText(modelFamilyBundleReportJson(selectedModelFamilyModules()), Charsets.UTF_8)
    }
}

tasks.register("printModelFamilyFixtureJson") {
    group = "help"
    description = "Print model-family fixture coverage as JSON for CI/release automation."
    dependsOn("writeModelFamilyFixtureReport")
    doLast {
        println(modelFamilyFixtureReportFile.get().asFile.readText(Charsets.UTF_8))
    }
}

tasks.register("printModelFamilyFixtureFingerprintJson") {
    group = "help"
    description = "Print compact model-family fixture fingerprints as JSON for CI/release automation."
    dependsOn("writeModelFamilyFixtureFingerprintReport")
    doLast {
        println(modelFamilyFixtureFingerprintReportFile.get().asFile.readText(Charsets.UTF_8))
    }
}

tasks.register("writeModelFamilyFixtureReport") {
    group = "help"
    description = "Write model-family fixture coverage as JSON for CI/release automation."

    val outputFile = modelFamilyFixtureReportFile
    outputs.file(outputFile)
    inputs.files(modelFamilyFixtureRootDirs())
    inputs.property("requiredModelFamilyFixtureTokens", requiredModelFamilyFixtureTokens.sorted().joinToString(","))
    inputs.property("requiredModelFamilyFixtures", requiredModelFamilyFixtures.sorted().joinToString(","))
    inputs.property("bundledModelFamilies", bundledModelFamilies.sorted().joinToString(","))
    inputs.property(
        "modelFamilyProfiles",
        modelFamilyModules.joinToString(",") { "${it.id}:${it.profile}:${it.path}" }
    )

    doLast {
        val file = outputFile.get().asFile
        file.parentFile.mkdirs()
        file.writeText(modelFamilyFixtureReportJson(), Charsets.UTF_8)
    }
}

tasks.register("writeModelFamilyFixtureFingerprintReport") {
    group = "help"
    description = "Write compact model-family fixture fingerprints as JSON for CI/release automation."

    val outputFile = modelFamilyFixtureFingerprintReportFile
    outputs.file(outputFile)
    inputs.files(modelFamilyFixtureRootDirs())
    inputs.property("requiredModelFamilyFixtureTokens", requiredModelFamilyFixtureTokens.sorted().joinToString(","))
    inputs.property("requiredModelFamilyFixtures", requiredModelFamilyFixtures.sorted().joinToString(","))
    inputs.property("bundledModelFamilies", bundledModelFamilies.sorted().joinToString(","))
    inputs.property(
        "modelFamilyProfiles",
        modelFamilyModules.joinToString(",") { "${it.id}:${it.profile}:${it.path}" }
    )

    doLast {
        val file = outputFile.get().asFile
        file.parentFile.mkdirs()
        file.writeText(modelFamilyFixtureFingerprintReportJson(), Charsets.UTF_8)
    }
}

val writeModelFamilyFixtureFingerprintLock = tasks.register("writeModelFamilyFixtureFingerprintLock") {
    group = "help"
    description = "Write a checked-in model-family fixture fingerprint lock for CI drift validation."
    dependsOn("writeModelFamilyFixtureFingerprintReport")
    dependsOn("validateModelFamilyFixtures")

    val outputFile = modelFamilyFixtureFingerprintLockFile
    outputs.file(outputFile)
    inputs.files(modelFamilyFixtureRootDirs())
    inputs.property("modelFamilyFixtureFingerprintLockPath", modelFamilyFixtureFingerprintLockPath)
    inputs.property("requiredModelFamilyFixtureTokens", requiredModelFamilyFixtureTokens.sorted().joinToString(","))
    inputs.property("requiredModelFamilyFixtures", requiredModelFamilyFixtures.sorted().joinToString(","))
    inputs.property("currentModelFamilyFixtureFingerprintLock", modelFamilyFixtureFingerprintLockInput())
    inputs.property(
        "modelFamilyProfiles",
        modelFamilyModules.joinToString(",") { "${it.id}:${it.profile}:${it.path}" }
    )

    doLast {
        val file = outputFile.asFile
        file.parentFile.mkdirs()
        file.writeText(modelFamilyFixtureFingerprintLockText(), Charsets.UTF_8)
        println("Wrote model-family fixture fingerprint lock: ${file.relativeTo(projectDir)}")
    }
}

val writeModelFamilyFixtureFingerprintLockDriftReport =
    tasks.register("writeModelFamilyFixtureFingerprintLockDriftReport") {
        group = "verification"
        description = "Write a JSON model-family fixture fingerprint lock drift report for CI artifacts."
        mustRunAfter(writeModelFamilyFixtureFingerprintLock)

        val inputFile = modelFamilyFixtureFingerprintLockFile
        val outputFile = modelFamilyFixtureFingerprintLockDriftReportFile
        inputs.file(inputFile)
        outputs.file(outputFile)
        inputs.files(modelFamilyFixtureRootDirs())
        inputs.property("modelFamilyFixtureFingerprintLockPath", modelFamilyFixtureFingerprintLockPath)
        inputs.property("requiredModelFamilyFixtureTokens", requiredModelFamilyFixtureTokens.sorted().joinToString(","))
        inputs.property("requiredModelFamilyFixtures", requiredModelFamilyFixtures.sorted().joinToString(","))
        inputs.property("currentModelFamilyFixtureFingerprintLock", modelFamilyFixtureFingerprintLockInput())

        doLast {
            val lockFile = inputFile.asFile
            val actual = if (lockFile.isFile) {
                Properties().also { properties ->
                    lockFile.inputStream().use(properties::load)
                }
            } else {
                null
            }
            val file = outputFile.get().asFile
            file.parentFile.mkdirs()
            file.writeText(
                modelFamilyFixtureFingerprintLockDriftReportJson(
                    lockFile,
                    modelFamilyFixtureFingerprintLockEntries(),
                    actual,
                    missingLock = !lockFile.isFile
                ),
                Charsets.UTF_8
            )
            println("Wrote model-family fixture fingerprint lock drift report: ${file.relativeTo(projectDir)}")
        }
    }

tasks.register("printModelFamilyFixtureFingerprintLockDriftJson") {
    group = "verification"
    description = "Print model-family fixture fingerprint lock drift as JSON for CI/release automation."
    dependsOn(writeModelFamilyFixtureFingerprintLockDriftReport)
    doLast {
        println(modelFamilyFixtureFingerprintLockDriftReportFile.get().asFile.readText(Charsets.UTF_8))
    }
}

val validateModelFamilyFixtureFingerprintLock = tasks.register("validateModelFamilyFixtureFingerprintLock") {
    group = "verification"
    description = "Validate model-family fixture fingerprints against a checked-in lock."
    mustRunAfter(writeModelFamilyFixtureFingerprintLock)
    dependsOn(writeModelFamilyFixtureFingerprintLockDriftReport)

    val inputFile = modelFamilyFixtureFingerprintLockFile
    inputs.file(inputFile)
    inputs.files(modelFamilyFixtureRootDirs())
    inputs.property("modelFamilyFixtureFingerprintLockPath", modelFamilyFixtureFingerprintLockPath)
    inputs.property("requiredModelFamilyFixtureTokens", requiredModelFamilyFixtureTokens.sorted().joinToString(","))
    inputs.property("requiredModelFamilyFixtures", requiredModelFamilyFixtures.sorted().joinToString(","))
    inputs.property("currentModelFamilyFixtureFingerprintLock", modelFamilyFixtureFingerprintLockInput())

    doLast {
        val file = inputFile.asFile
        if (!file.isFile) {
            throw GradleException(
                "Model-family fixture fingerprint lock validation failed: missing ${file.relativeTo(projectDir)}. " +
                        "Run :ui:gollek-cli:writeModelFamilyFixtureFingerprintLock with the intended " +
                        "-Pgollek.requiredModelFamilyFixtures selector, then commit the lock. " +
                        "Report: ${modelFamilyFixtureFingerprintLockDriftReportFile.get().asFile.relativeTo(projectDir)}"
            )
        }

        val expected = modelFamilyFixtureFingerprintLockEntries()
        val actual = Properties()
        file.inputStream().use(actual::load)
        val driftEntries = modelFamilyFixtureFingerprintLockDriftEntries(expected, actual)
        val problems = modelFamilyBundleLockKeyProblems(driftEntries)

        if (problems.isNotEmpty()) {
            val fixtureSummaries = modelFamilyBundleLockFixtureDriftSummaries(expected, actual)
            throw GradleException(
                "Model-family fixture fingerprint lock drift detected in ${file.relativeTo(projectDir)}:\n" +
                        (
                                if (fixtureSummaries.isEmpty()) {
                                    ""
                                } else {
                                    "Fixture fingerprint drift:\n" +
                                            fixtureSummaries.joinToString("\n") { "  - $it" } +
                                            "\n"
                                }
                                ) +
                        "Raw lock key drift:\n" +
                        problems.joinToString("\n") { "  - $it" } +
                        "\nReport: ${modelFamilyFixtureFingerprintLockDriftReportFile.get().asFile.relativeTo(projectDir)}" +
                        "\nIf this change is intentional, rerun :ui:gollek-cli:writeModelFamilyFixtureFingerprintLock " +
                        "with the same fixture selectors and review the diff."
            )
        }
    }
}

val validateModelFamilyFixtures = tasks.register("validateModelFamilyFixtures") {
    group = "verification"
    description = "Validate required model-family fixture coverage for config/tokenizer drift checks."
    dependsOn("writeModelFamilyFixtureReport")

    inputs.files(modelFamilyFixtureRootDirs())
    inputs.property("requiredModelFamilyFixtureTokens", requiredModelFamilyFixtureTokens.sorted().joinToString(","))
    inputs.property("requiredModelFamilyFixtures", requiredModelFamilyFixtures.sorted().joinToString(","))

    doLast {
        val requiredProblems = modelFamilyFixtureInventories()
            .filter { it.required && !it.passed }
            .flatMap { inventory ->
                inventory.problems.map { problem -> "${inventory.module.id}: $problem (${inventory.fixturePath})" }
            }

        if (requiredProblems.isNotEmpty()) {
            throw GradleException(
                "Model-family fixture validation failed:\n" +
                        requiredProblems.joinToString("\n") { "  - $it" } +
                        "\nReport: ${modelFamilyFixtureReportFile.get().asFile.relativeTo(projectDir)}"
            )
        }
    }
}

val validateModelFamilyBundle = tasks.register("validateModelFamilyBundle") {
    group = "verification"
    description = "Validate model-family Gradle metadata, plugin descriptors, and ServiceLoader entries."

    val pluginJsonFiles = existingModelFamilyModules().map { module ->
        modelFamilyProjectDir(module).resolve("src/main/resources/plugin.json")
    }
    val pluginSourceDirs = existingModelFamilyModules().map { module ->
        modelFamilyProjectDir(module).resolve("src/main/java")
    }
    inputs.files(pluginJsonFiles)
    inputs.files(pluginSourceDirs)
    inputs.property("pendingModelFamilyTokenizerMetadata", pendingModelFamilyTokenizerMetadataInput())

    doLast {
        val modelFamilyService = "META-INF/services/tech.kayys.gollek.spi.model.ModelFamilyPlugin"
        val gollekPluginService = "META-INF/services/tech.kayys.gollek.spi.plugin.GollekPlugin"
        val problems = mutableListOf<String>()

        for (module in modelFamilyModules) {
            if (modelFamilyProject(module) == null) {
                if (module.id in bundledModelFamilies) {
                    problems += "${module.id}: selected model-family project ${module.path} is not included"
                }
                continue
            }

            val moduleProjectDir = modelFamilyProjectDir(module)
            val resourcesDir = moduleProjectDir.resolve("src/main/resources")
            val pluginJson = resourcesDir.resolve("plugin.json")

            if (!pluginJson.isFile) {
                problems += "${module.id}: missing plugin.json at ${pluginJson.relativeTo(rootDir)}"
                continue
            }

            val json = pluginJson.readText()
            val descriptorId = jsonStringValue(json, "id")
            val mainClass = jsonStringValue(json, "mainClass")
            val bundleProfile = jsonStringValue(json, "bundleProfile")
            val tokenizerKinds = modelFamilyPluginDescriptorTokenizerKinds(module.path)
            val tokenizerMetadataPending = modelFamilyPluginDescriptorTokenizerMetadataPending(module.path)
            val tokenizerMetadataPendingReason = modelFamilyPluginDescriptorTokenizerMetadataPendingReason(module.path)

            if (descriptorId != "model-family/${module.id}") {
                problems += "${module.id}: plugin.json id is '$descriptorId', expected 'model-family/${module.id}'"
            }
            if (bundleProfile != module.profile) {
                problems += "${module.id}: plugin.json bundleProfile is '$bundleProfile', expected '${module.profile}'"
            }
            if (tokenizerMetadataPending && tokenizerMetadataPendingReason.isBlank()) {
                problems += "${module.id}: plugin.json tokenizerMetadataStatus is pending but " +
                        "tokenizerMetadataPendingReason is missing"
            }
            if (!tokenizerMetadataPending && tokenizerMetadataPendingReason.isNotBlank()) {
                problems += "${module.id}: plugin.json tokenizerMetadataPendingReason is set but " +
                        "tokenizerMetadataStatus is not pending"
            }
            if (tokenizerKinds.isEmpty() && module.id !in pendingModelFamilyTokenizerMetadata) {
                problems += "${module.id}: plugin.json is missing properties.tokenizerKind/tokenizerKinds"
            }
            if (tokenizerKinds.isNotEmpty() && module.id in pendingModelFamilyTokenizerMetadata) {
                problems += "${module.id}: tokenizer metadata is now present; clear " +
                        "plugin.json tokenizerMetadataStatus=pending"
            }
            if (mainClass.isNullOrBlank()) {
                problems += "${module.id}: plugin.json mainClass is missing"
                continue
            }

            for (servicePath in listOf(modelFamilyService, gollekPluginService)) {
                val serviceFile = resourcesDir.resolve(servicePath)
                if (!serviceFile.isFile) {
                    problems += "${module.id}: missing ServiceLoader file $servicePath"
                    continue
                }
                val implementations = serviceFile.readLines()
                    .map { it.substringBefore("#").trim() }
                    .filter { it.isNotBlank() }
                if (mainClass !in implementations) {
                    problems += "${module.id}: $servicePath does not list $mainClass"
                }
            }

            validateScopedDirectSafetensorMetadata(
                module,
                moduleProjectDir.resolve("src/main/java"),
                problems
            )
        }

        if (problems.isNotEmpty()) {
            throw GradleException(
                "Model-family bundle metadata validation failed:\n" +
                        problems.joinToString("\n") { "  - $it" }
            )
        }
    }
}

val validateModelFamilyModuleCatalog = tasks.register("validateModelFamilyModuleCatalog") {
    group = "verification"
    description = "Validate that attachable model-family Gradle projects and CLI selector catalog stay in sync."
    dependsOn("writeModelFamilyModuleCatalogReport")

    inputs.property("catalogModelFamilyProjectPaths", catalogModelFamilyProjectPaths().sorted().joinToString(","))
    inputs.property("includedModelFamilyProjectPaths", includedModelFamilyProjectPaths().sorted().joinToString(","))
    inputs.property("pendingModelFamilyTokenizerMetadata", pendingModelFamilyTokenizerMetadataInput())
    inputs.property(
        "includedCatalogedModelFamilyProjectPaths",
        includedCatalogedModelFamilyProjectPaths().sorted().joinToString(",")
    )
    inputs.files(includedModelFamilyProjectPaths().flatMap { path ->
        listOf(
            modelFamilyProjectResourceFile(path, modelFamilyPluginDescriptorRelativePath),
            modelFamilyProjectResourceFile(path, modelFamilyServiceDescriptorRelativePath),
            modelFamilyProjectResourceFile(path, gollekPluginServiceDescriptorRelativePath)
        )
    })

    doLast {
        val missingProjects = modelFamilyModuleCatalogMissingProjects()
        val uncatalogedProjects = modelFamilyModuleCatalogUncatalogedProjects()
        val serviceIncompleteProjects = modelFamilyModuleCatalogServiceIncompleteProjects()
        val tokenizerMissingFamilies = modelFamilyModuleCatalogTokenizerMetadataMissingFamilies()
        val productionReadinessPendingFamilies = modelFamilyModuleCatalogProductionReadinessPendingFamilies()
        val directSafetensorPendingFamilies = modelFamilyModuleCatalogDirectSafetensorPendingFamilies()
        val stalePendingFamilies = modelFamilyModules
            .filter { module -> modelFamilyProject(module) != null }
            .filter { module -> module.id in pendingModelFamilyTokenizerMetadata }
            .filter { module -> modelFamilyPluginDescriptorHasTokenizerMetadata(module.path) }
            .map { it.id }
        val moduleById = modelFamilyModules.associateBy { it.id }

        val problems = mutableListOf<String>()
        if (missingProjects.isNotEmpty()) {
            problems += "catalog entries are not included as Gradle projects: " +
                    missingProjects.sorted().joinToString(", ")
        }
        if (uncatalogedProjects.isNotEmpty()) {
            problems += "included model-family plugin projects are missing from the CLI catalog: " +
                    uncatalogedProjects.sorted().joinToString(", ")
        }
        if (serviceIncompleteProjects.isNotEmpty()) {
            problems += "cataloged model-family plugin projects are missing plugin service descriptors: " +
                    serviceIncompleteProjects.sorted().joinToString(", ")
        }
        if (tokenizerMissingFamilies.isNotEmpty()) {
            problems += "cataloged model-family plugin projects are missing tokenizer metadata: " +
                    tokenizerMissingFamilies.sorted().joinToString(", ")
        }
        if (stalePendingFamilies.isNotEmpty()) {
            problems += "model-family tokenizer metadata pending exceptions are stale: " +
                    stalePendingFamilies.sorted()
                        .joinToString(", ") { familyId ->
                            "$familyId (${modelFamilyTokenizerMetadataPendingReason(familyId)})"
                        }
        }
        if (productionReadinessPendingFamilies.isNotEmpty()) {
            problems += "production candidate model-family plugins are not production-ready: " +
                    productionReadinessPendingFamilies.sorted()
                        .joinToString(", ") { familyId ->
                            val readiness = moduleById[familyId]?.let(::modelFamilyModuleReadiness) ?: "unknown"
                            "$familyId ($readiness)"
                        }
        }
        if (directSafetensorPendingFamilies.isNotEmpty()) {
            problems += "direct SafeTensor model-family plugins are not direct-runtime-ready: " +
                    directSafetensorPendingFamilies.sorted()
                        .joinToString(", ") { familyId ->
                            val readiness = moduleById[familyId]?.let(::modelFamilyModuleReadiness) ?: "unknown"
                            "$familyId ($readiness)"
                        }
        }

        if (problems.isNotEmpty()) {
            throw GradleException(
                "Model-family module catalog validation failed:\n" +
                        problems.joinToString("\n") { "  - $it" } +
                        "\nReport: ${modelFamilyModuleCatalogReportFile.get().asFile.relativeTo(projectDir)}" +
                        "\nUpdate settings.gradle.kts auto-discovery or ui/gollek-cli modelFamilyModules."
            )
        }
    }
}

val validateExtensionAvailabilityProviders = tasks.register("validateExtensionAvailabilityProviders") {
    group = "verification"
    description = "Validate extension availability provider ServiceLoader descriptors."

    val servicePath = "META-INF/services/tech.kayys.gollek.plugin.core.ExtensionAvailabilityProvider"
    val serviceFile = projectDir.resolve("src/main/resources/$servicePath")
    val sourceRoot = projectDir.resolve("src/main/java")
    val expectedProviders = listOf(
        "tech.kayys.gollek.cli.util.SulingAudioExtensionAvailabilityProvider",
        "tech.kayys.gollek.cli.util.Gemma4UnifiedRuntimeAvailabilityProvider"
    )

    inputs.file(serviceFile)
    inputs.files(fileTree(sourceRoot) {
        include("**/*ExtensionAvailabilityProvider.java")
    })
    inputs.property("expectedExtensionAvailabilityProviders", expectedProviders.joinToString(","))

    doLast {
        val problems = mutableListOf<String>()

        if (!serviceFile.isFile) {
            problems += "missing ServiceLoader file $servicePath"
        } else {
            val implementations = serviceFile.readLines()
                .map { it.substringBefore("#").trim() }
                .filter { it.isNotBlank() }
            val duplicates = implementations.groupingBy { it }
                .eachCount()
                .filterValues { it > 1 }
                .keys
                .sorted()

            if (implementations.isEmpty()) {
                problems += "$servicePath does not list any providers"
            }
            for (duplicate in duplicates) {
                problems += "$servicePath lists duplicate provider $duplicate"
            }
            for (provider in expectedProviders) {
                if (provider !in implementations) {
                    problems += "$servicePath does not list required provider $provider"
                }
            }
            for (implementation in implementations) {
                val sourceFile = sourceRoot.resolve(implementation.replace('.', File.separatorChar) + ".java")
                if (!sourceFile.isFile) {
                    problems += "$servicePath lists $implementation, but source file is missing"
                    continue
                }
                val source = sourceFile.readText()
                if (!source.contains("implements ExtensionAvailabilityProvider")) {
                    problems += "$implementation must implement ExtensionAvailabilityProvider"
                }
                if (!source.contains("ExtensionAvailability availability()")) {
                    problems += "$implementation must expose an ExtensionAvailability availability() method"
                }
            }
        }

        if (problems.isNotEmpty()) {
            throw GradleException(
                "Extension availability provider validation failed:\n" +
                        problems.joinToString("\n") { "  - $it" }
            )
        }
    }
}

val pluginClasspath = providers.gradleProperty("gollek.pluginClasspath").orElse("")
val extensionPluginClasspath = providers.gradleProperty("gollek.extensionPluginClasspath").orElse("")
val pluginDirs = providers.gradleProperty("gollek.pluginDirs").orElse("")
val extensionPluginDirs = providers.gradleProperty("gollek.extensionPluginDirs").orElse("")

val validateExtensionAvailabilityProviderContracts = tasks.register<JavaExec>(
    "validateExtensionAvailabilityProviderContracts"
) {
    group = "verification"
    description = "Run packaged extension availability providers through the plugin-core contract gate."
    dependsOn(tasks.named("classes"))
    classpath = sourceSets.named("main").get().runtimeClasspath
    mainClass.set("tech.kayys.gollek.cli.util.ExtensionAvailabilityContractCheck")
    inputs.property("pluginDirs", pluginDirs)
    inputs.property("extensionPluginClasspath", extensionPluginClasspath)
    inputs.property("extensionPluginDirs", extensionPluginDirs)
    pluginPathInputFiles("extensionPluginClasspathFiles", pluginClasspath, extensionPluginClasspath)
    pluginPathInputFiles("extensionPluginDirectoryFiles", pluginDirs, extensionPluginDirs)
    doFirst {
        args(extensionPluginClasspathArgs())
    }
}

val extensionAvailabilityGateReportFile =
    layout.buildDirectory.file("reports/gollek/extension-availability-gate.json")

val writeExtensionAvailabilityGateReport = tasks.register<JavaExec>("writeExtensionAvailabilityGateReport") {
    group = "verification"
    description = "Write the packaged extension availability release-gate report for CI archival."
    dependsOn(tasks.named("classes"))
    classpath = sourceSets.named("main").get().runtimeClasspath
    mainClass.set("tech.kayys.gollek.cli.util.ExtensionAvailabilityGateReportWriter")
    args(extensionAvailabilityGateReportFile.get().asFile.absolutePath)
    inputs.property("pluginDirs", pluginDirs)
    inputs.property("extensionPluginClasspath", extensionPluginClasspath)
    inputs.property("extensionPluginDirs", extensionPluginDirs)
    pluginPathInputFiles("extensionPluginClasspathFiles", pluginClasspath, extensionPluginClasspath)
    pluginPathInputFiles("extensionPluginDirectoryFiles", pluginDirs, extensionPluginDirs)
    outputs.file(extensionAvailabilityGateReportFile)
    doFirst {
        args(extensionPluginClasspathArgs())
    }
}

val modelFamilyPluginClasspath = providers.gradleProperty("gollek.modelFamilyPluginClasspath").orElse("")
val modelFamilyPluginDirs = providers.gradleProperty("gollek.modelFamilyPluginDirs").orElse("")

fun pluginPathEntries(vararg values: Provider<String>): List<String> {
    return values.asSequence()
        .flatMap { value -> value.get().split(File.pathSeparator).asSequence() }
        .map(String::trim)
        .filter(String::isNotEmpty)
        .map { rootProject.file(it).absolutePath }
        .distinct()
        .toList()
}

fun optionPathArgs(optionName: String, vararg values: Provider<String>): List<String> {
    return pluginPathEntries(*values)
        .flatMap { path -> listOf(optionName, path) }
}

fun pluginClasspathArgs(vararg values: Provider<String>): List<String> {
    return pluginPathEntries(*values)
}

fun pluginDirArgs(vararg values: Provider<String>): List<String> {
    return optionPathArgs("--plugin-dir", *values)
}

fun JavaExec.pluginPathInputFiles(propertyName: String, vararg values: Provider<String>) {
    inputs.files(providers.provider {
        pluginPathEntries(*values).map { rootProject.file(it) }
    })
        .withPropertyName(propertyName)
        .withPathSensitivity(PathSensitivity.RELATIVE)
        .optional()
}

fun extensionPluginClasspathArgs(): List<String> {
    return pluginClasspathArgs(pluginClasspath, extensionPluginClasspath) +
            pluginDirArgs(pluginDirs, extensionPluginDirs)
}

fun modelFamilyPluginClasspathArgs(): List<String> {
    return pluginClasspathArgs(pluginClasspath, modelFamilyPluginClasspath) +
            pluginDirArgs(pluginDirs, modelFamilyPluginDirs)
}

fun combinedPluginClasspathArgs(): List<String> {
    return pluginClasspathArgs(pluginClasspath, extensionPluginClasspath, modelFamilyPluginClasspath) +
            pluginDirArgs(pluginDirs, extensionPluginDirs, modelFamilyPluginDirs)
}

val validateModelFamilyBundleGate = tasks.register<JavaExec>("validateModelFamilyBundleGate") {
    group = "verification"
    description = "Run packaged model-family plugins through the bundle availability and contract release gate."
    dependsOn(tasks.named("classes"))
    classpath = sourceSets.named("main").get().runtimeClasspath
    mainClass.set("tech.kayys.gollek.cli.util.ModelFamilyBundleGateCheck")
    inputs.property("pluginDirs", pluginDirs)
    inputs.property("modelFamilyPluginClasspath", modelFamilyPluginClasspath)
    inputs.property("modelFamilyPluginDirs", modelFamilyPluginDirs)
    pluginPathInputFiles("modelFamilyPluginClasspathFiles", pluginClasspath, modelFamilyPluginClasspath)
    pluginPathInputFiles("modelFamilyPluginDirectoryFiles", pluginDirs, modelFamilyPluginDirs)
    doFirst {
        args(modelFamilyPluginClasspathArgs())
    }
}

val modelFamilyBundleGateReportFile =
    layout.buildDirectory.file("reports/gollek/model-family-bundle-gate.json")

val writeModelFamilyBundleGateReport = tasks.register<JavaExec>("writeModelFamilyBundleGateReport") {
    group = "verification"
    description = "Write the packaged model-family bundle release-gate report for CI archival."
    dependsOn(tasks.named("classes"))
    classpath = sourceSets.named("main").get().runtimeClasspath
    mainClass.set("tech.kayys.gollek.cli.util.ModelFamilyBundleGateReportWriter")
    args(modelFamilyBundleGateReportFile.get().asFile.absolutePath)
    inputs.property("pluginDirs", pluginDirs)
    inputs.property("modelFamilyPluginClasspath", modelFamilyPluginClasspath)
    inputs.property("modelFamilyPluginDirs", modelFamilyPluginDirs)
    pluginPathInputFiles("modelFamilyPluginClasspathFiles", pluginClasspath, modelFamilyPluginClasspath)
    pluginPathInputFiles("modelFamilyPluginDirectoryFiles", pluginDirs, modelFamilyPluginDirs)
    outputs.file(modelFamilyBundleGateReportFile)
    doFirst {
        args(modelFamilyPluginClasspathArgs())
    }
}

val pluginGatesReportFile =
    layout.buildDirectory.file("reports/gollek/plugin-gates.json")

val validatePluginGates = tasks.register<JavaExec>("validatePluginGates") {
    group = "verification"
    description = "Run extension and model-family plugin release gates as one combined CI check."
    dependsOn(tasks.named("classes"))
    classpath = sourceSets.named("main").get().runtimeClasspath
    mainClass.set("tech.kayys.gollek.cli.util.PluginGatesCheck")
    inputs.property("pluginClasspath", pluginClasspath)
    inputs.property("pluginDirs", pluginDirs)
    inputs.property("extensionPluginClasspath", extensionPluginClasspath)
    inputs.property("extensionPluginDirs", extensionPluginDirs)
    inputs.property("modelFamilyPluginClasspath", modelFamilyPluginClasspath)
    inputs.property("modelFamilyPluginDirs", modelFamilyPluginDirs)
    pluginPathInputFiles(
        "combinedPluginClasspathFiles",
        pluginClasspath,
        extensionPluginClasspath,
        modelFamilyPluginClasspath
    )
    pluginPathInputFiles(
        "combinedPluginDirectoryFiles",
        pluginDirs,
        extensionPluginDirs,
        modelFamilyPluginDirs
    )
    doFirst {
        args(combinedPluginClasspathArgs())
    }
}

val writePluginGatesReport = tasks.register<JavaExec>("writePluginGatesReport") {
    group = "verification"
    description = "Write a combined extension and model-family plugin release-gate report for CI archival."
    dependsOn(tasks.named("classes"))
    classpath = sourceSets.named("main").get().runtimeClasspath
    mainClass.set("tech.kayys.gollek.cli.util.PluginGatesReportWriter")
    args(pluginGatesReportFile.get().asFile.absolutePath)
    inputs.property("pluginClasspath", pluginClasspath)
    inputs.property("pluginDirs", pluginDirs)
    inputs.property("extensionPluginClasspath", extensionPluginClasspath)
    inputs.property("extensionPluginDirs", extensionPluginDirs)
    inputs.property("modelFamilyPluginClasspath", modelFamilyPluginClasspath)
    inputs.property("modelFamilyPluginDirs", modelFamilyPluginDirs)
    pluginPathInputFiles(
        "combinedPluginClasspathFiles",
        pluginClasspath,
        extensionPluginClasspath,
        modelFamilyPluginClasspath
    )
    pluginPathInputFiles(
        "combinedPluginDirectoryFiles",
        pluginDirs,
        extensionPluginDirs,
        modelFamilyPluginDirs
    )
    outputs.file(pluginGatesReportFile)
    doFirst {
        args(combinedPluginClasspathArgs())
    }
}

val validateModelFamilyBundleManifest = tasks.register("validateModelFamilyBundleManifest") {
    group = "verification"
    description = "Validate generated model-family bundle manifest consistency."
    dependsOn(generateModelFamilyBundleManifest)

    val manifestFile = modelFamilyBundleManifestFile
    inputs.file(manifestFile)
    inputs.property("requestedModelFamilyTokens", requestedModelFamilyTokens.sorted().joinToString(","))
    inputs.property("requestedModelFamilySelectorSource", requestedModelFamilySelectorSource)
    inputs.property("explicitModelFamilyTokens", explicitModelFamilyTokens.sorted().joinToString(","))
    inputs.property("presetModelFamilyTokens", presetModelFamilyTokens.sorted().joinToString(","))
    inputs.property("defaultModelFamilyTokens", defaultModelFamilyTokens.sorted().joinToString(","))
    inputs.property("modelFamilyPolicySource", modelFamilyPolicySource)
    inputs.property("presetRequiredModelFamilyTokens", presetRequiredModelFamilyTokens.sorted().joinToString(","))
    inputs.property("presetForbiddenModelFamilyTokens", presetForbiddenModelFamilyTokens.sorted().joinToString(","))
    inputs.property("presetRequiredModelFamilyAliasTokens", presetRequiredModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("presetForbiddenModelFamilyAliasTokens", presetForbiddenModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("explicitRequiredModelFamilyTokens", explicitRequiredModelFamilyTokens.sorted().joinToString(","))
    inputs.property("explicitForbiddenModelFamilyTokens", explicitForbiddenModelFamilyTokens.sorted().joinToString(","))
    inputs.property("explicitRequiredModelFamilyAliasTokens", explicitRequiredModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("explicitForbiddenModelFamilyAliasTokens", explicitForbiddenModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("requestedModelFamilyBundlePreset", requestedModelFamilyBundlePreset?.id ?: "")
    inputs.property("modelFamilyBundlePresets", modelFamilyBundlePresetInput())
    inputs.property("bundledModelFamilies", bundledModelFamilies.sorted().joinToString(","))
    inputs.property("requiredModelFamilies", requiredModelFamilies.sorted().joinToString(","))
    inputs.property("forbiddenModelFamilies", forbiddenModelFamilies.sorted().joinToString(","))
    inputs.property("requiredModelFamilyAliases", requiredModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("forbiddenModelFamilyAliases", forbiddenModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property(
        "productionReadinessPendingFamilies",
        modelFamilyModuleCatalogProductionReadinessPendingFamilies().sorted().joinToString(",")
    )
    inputs.property(
        "directSafetensorPendingFamilies",
        modelFamilyModuleCatalogDirectSafetensorPendingFamilies().sorted().joinToString(",")
    )
    inputs.property("pendingModelFamilyTokenizerMetadata", pendingModelFamilyTokenizerMetadataInput())
    inputs.property("manifestSchemaVersion", modelFamilyBundleManifestSchemaVersion)
    inputs.files(modelFamilyFixtureRootDirs())
    inputs.property("requiredModelFamilyFixtureTokens", requiredModelFamilyFixtureTokens.sorted().joinToString(","))
    inputs.property("requiredModelFamilyFixtures", requiredModelFamilyFixtures.sorted().joinToString(","))

    doLast {
        val selected = selectedModelFamilyModules()
        val missingRequired = modelFamilyPolicyMissingRequired()
        val selectedForbidden = modelFamilyPolicySelectedForbidden()
        val missingRequiredAliases = modelFamilyPolicyMissingRequiredAliases()
        val selectedForbiddenAliases = modelFamilyPolicySelectedForbiddenAliases()
        val policyViolationCount = modelFamilyPolicyViolationCount(
            missingRequired,
            selectedForbidden,
            missingRequiredAliases,
            selectedForbiddenAliases
        )
        val fixtureSnapshot = modelFamilyFixtureLockSnapshot()
        val productionReadinessPendingFamilies = modelFamilyModuleCatalogProductionReadinessPendingFamilies()
        val directSafetensorPendingFamilies = modelFamilyModuleCatalogDirectSafetensorPendingFamilies()
        val manifest = manifestFile.get().asFile
        val problems = mutableListOf<String>()

        if (!manifest.isFile) {
            throw GradleException(
                "Model-family bundle manifest validation failed: missing ${manifest.relativeTo(projectDir)}"
            )
        }

        val properties = Properties()
        manifest.inputStream().use(properties::load)

        fun expectProperty(key: String, expected: String) {
            val actual = properties.getProperty(key)
            if (actual != expected) {
                problems += "$key is '$actual', expected '$expected'"
            }
        }

        fun expectCsv(key: String, expected: List<String>) {
            val actual = csvProperty(properties, key)
            if (actual != expected) {
                problems += "$key is '${actual.joinToString(",")}', expected '${expected.joinToString(",")}'"
            }
        }

        expectProperty("schemaVersion", modelFamilyBundleManifestSchemaVersion)
        expectProperty("bundleFingerprint", modelFamilyBundleFingerprint(selected))
        expectProperty("bundlePreset", requestedModelFamilyBundlePreset?.id ?: "")
        expectProperty("selectorSource", requestedModelFamilySelectorSource)
        expectCsv("explicitSelectors", explicitModelFamilyTokens.sorted())
        expectCsv("presetSelectors", presetModelFamilyTokens.sorted())
        expectCsv("defaultSelectors", defaultModelFamilyTokens.sorted())
        expectProperty("policySource", modelFamilyPolicySource)
        expectCsv("presetRequiredFamilies", presetRequiredModelFamilyTokens.sorted())
        expectCsv("presetForbiddenFamilies", presetForbiddenModelFamilyTokens.sorted())
        expectCsv("presetRequiredAliases", presetRequiredModelFamilyAliasTokens.sorted())
        expectCsv("presetForbiddenAliases", presetForbiddenModelFamilyAliasTokens.sorted())
        expectCsv("explicitRequiredFamilies", explicitRequiredModelFamilyTokens.sorted())
        expectCsv("explicitForbiddenFamilies", explicitForbiddenModelFamilyTokens.sorted())
        expectCsv("explicitRequiredAliases", explicitRequiredModelFamilyAliasTokens.sorted())
        expectCsv("explicitForbiddenAliases", explicitForbiddenModelFamilyAliasTokens.sorted())
        expectProperty("requiresDirectSafetensorRuntime", requiresDirectSafetensorRuntime.toString())
        expectProperty("productionReadinessPassed", productionReadinessPendingFamilies.isEmpty().toString())
        expectProperty("productionReadinessPendingCount", productionReadinessPendingFamilies.size.toString())
        expectCsv("productionReadinessPendingFamilies", productionReadinessPendingFamilies.sorted())
        expectProperty("directSafetensorReadinessPassed", directSafetensorPendingFamilies.isEmpty().toString())
        expectProperty("directSafetensorPendingCount", directSafetensorPendingFamilies.size.toString())
        expectCsv("directSafetensorPendingFamilies", directSafetensorPendingFamilies.sorted())
        expectProperty("selectors", requestedModelFamilyTokens.sorted().joinToString(","))
        expectCsv("requiredFamilies", requiredModelFamilies.sorted())
        expectCsv("forbiddenFamilies", forbiddenModelFamilies.sorted())
        expectCsv("requiredAliases", requiredModelFamilyAliasTokens.sorted())
        expectCsv("forbiddenAliases", forbiddenModelFamilyAliasTokens.sorted())
        expectProperty("policyPassed", (policyViolationCount == 0).toString())
        expectProperty("policyViolationCount", policyViolationCount.toString())
        expectCsv("missingRequiredFamilies", missingRequired.sorted())
        expectCsv("selectedForbiddenFamilies", selectedForbidden.sorted())
        expectCsv("missingRequiredAliases", missingRequiredAliases.keys.sorted())
        expectCsv("selectedForbiddenAliases", selectedForbiddenAliases.keys.sorted())
        expectCsv("fixtureRequiredSelectors", fixtureSnapshot.requiredSelectors)
        expectCsv("fixtureRequiredFamilies", fixtureSnapshot.requiredFamilies)
        expectProperty("fixturePassed", fixtureSnapshot.passed.toString())
        expectProperty("fixtureRequiredFingerprint", fixtureSnapshot.requiredFingerprint)
        expectProperty("fixtureInventoryFingerprint", fixtureSnapshot.inventoryFingerprint)
        expectProperty("fixtureAvailableFamilyCount", fixtureSnapshot.availableFamilyCount.toString())
        expectProperty("fixtureFamilyCount", fixtureSnapshot.fixtureFamilyCount.toString())
        expectProperty("fixtureRequiredFamilyCount", fixtureSnapshot.requiredFamilyCount.toString())
        expectProperty("fixtureRequiredPassedCount", fixtureSnapshot.requiredPassedCount.toString())
        expectProperty("fixtureMissingRequiredCount", fixtureSnapshot.missingRequiredCount.toString())
        expectProperty("fixtureProblemFamilyCount", fixtureSnapshot.problemFamilyCount.toString())
        expectCsv("fixtureMissingRequiredFamilies", fixtureSnapshot.missingRequiredFamilies)
        expectCsv("fixtureProblemFamilies", fixtureSnapshot.problemFamilies)
        for ((aliasId, missing) in missingRequiredAliases) {
            expectCsv("missingRequiredAlias.${aliasId}.families", missing.sorted())
        }
        for ((aliasId, selectedFamilies) in selectedForbiddenAliases) {
            expectCsv("selectedForbiddenAlias.${aliasId}.families", selectedFamilies.sorted())
        }
        expectProperty("detached", selected.isEmpty().toString())
        expectProperty("familyCount", selected.size.toString())
        expectCsv("families", selected.map { it.id })
        expectCsv("profiles", selected.map { it.profile }.distinct().sorted())
        expectCsv("availableFamilies", modelFamilyModules.map { it.id }.sorted())
        expectCsv("availableProfiles", modelFamilyModules.map { it.profile }.distinct().sorted())
        expectCsv("availableSelectors", availableModelFamilySelectors)
        expectCsv("tokenizerMetadataPendingFamilies", pendingModelFamilyTokenizerMetadata.sorted())
        expectCsv("availableBundlePresets", modelFamilyBundlePresetIds.sorted())
        expectCsv("bundlePresets", modelFamilyBundlePresets.map { it.id }.sorted())
        expectCsv("bundleAliases", modelFamilyBundleAliases.map { it.id }.sorted())

        for (preset in modelFamilyBundlePresets) {
            val validation = modelFamilyBundlePresetValidation(preset)
            expectProperty("bundlePreset.${preset.id}.description", preset.description)
            expectCsv("bundlePreset.${preset.id}.selectors", preset.selectors.sorted())
            expectCsv("bundlePreset.${preset.id}.requiredFamilies", preset.requiredFamilies.sorted())
            expectCsv("bundlePreset.${preset.id}.forbiddenFamilies", preset.forbiddenFamilies.sorted())
            expectCsv("bundlePreset.${preset.id}.requiredAliases", preset.requiredAliases.sorted())
            expectCsv("bundlePreset.${preset.id}.forbiddenAliases", preset.forbiddenAliases.sorted())
            expectCsv("bundlePreset.${preset.id}.selectedFamilies", validation.selectedFamilyIds.sorted())
            expectProperty("bundlePreset.${preset.id}.selectedCount", validation.selectedFamilyIds.size.toString())
            expectProperty(
                "bundlePreset.${preset.id}.productionTokenizerMetadataRequired",
                validation.productionTokenizerMetadataRequired.toString()
            )
            expectProperty(
                "bundlePreset.${preset.id}.productionTokenizerMetadataReady",
                validation.productionTokenizerMetadataReady.toString()
            )
            expectProperty(
                "bundlePreset.${preset.id}.productionSafetyPassed",
                validation.productionSafetyPassed.toString()
            )
            expectProperty(
                "bundlePreset.${preset.id}.productionSafetyViolationCount",
                validation.productionSafetyViolationCount.toString()
            )
            expectCsv(
                "bundlePreset.${preset.id}.pendingTokenizerFamilies",
                validation.pendingTokenizerFamilyIds.sorted()
            )
            for ((familyId, reason) in validation.pendingTokenizerReasons) {
                expectProperty("bundlePreset.${preset.id}.pendingTokenizerFamily.${familyId}.reason", reason)
            }
            expectProperty("bundlePreset.${preset.id}.policyPassed", validation.passed.toString())
            expectProperty("bundlePreset.${preset.id}.policyViolationCount", validation.violationCount.toString())
            expectCsv("bundlePreset.${preset.id}.missingRequiredFamilies", validation.missingRequiredFamilyIds.sorted())
            expectProperty(
                "bundlePreset.${preset.id}.missingRequiredCount",
                validation.missingRequiredFamilyIds.size.toString()
            )
            expectCsv(
                "bundlePreset.${preset.id}.selectedForbiddenFamilies",
                validation.selectedForbiddenFamilyIds.sorted()
            )
            expectProperty(
                "bundlePreset.${preset.id}.selectedForbiddenCount",
                validation.selectedForbiddenFamilyIds.size.toString()
            )
            expectCsv(
                "bundlePreset.${preset.id}.missingRequiredAliases",
                validation.missingRequiredAliases.keys.sorted()
            )
            expectProperty(
                "bundlePreset.${preset.id}.missingRequiredAliasCount",
                validation.missingRequiredAliases.size.toString()
            )
            expectCsv(
                "bundlePreset.${preset.id}.selectedForbiddenAliases",
                validation.selectedForbiddenAliases.keys.sorted()
            )
            expectProperty(
                "bundlePreset.${preset.id}.selectedForbiddenAliasCount",
                validation.selectedForbiddenAliases.size.toString()
            )
            for ((aliasId, missing) in validation.missingRequiredAliases) {
                expectCsv("bundlePreset.${preset.id}.missingRequiredAlias.${aliasId}.families", missing.sorted())
            }
            for ((aliasId, selectedFamilies) in validation.selectedForbiddenAliases) {
                expectCsv(
                    "bundlePreset.${preset.id}.selectedForbiddenAlias.${aliasId}.families",
                    selectedFamilies.sorted()
                )
            }
        }

        for (alias in modelFamilyBundleAliases) {
            expectProperty("bundleAlias.${alias.id}.description", alias.description)
            expectCsv("bundleAlias.${alias.id}.families", alias.familyIds.sorted())
            expectProperty("bundleAlias.${alias.id}.familyCount", alias.familyIds.size.toString())
        }

        for (module in modelFamilyModules) {
            expectProperty("family.${module.id}.selected", (module.id in bundledModelFamilies).toString())
            expectProperty("family.${module.id}.profile", module.profile)
            expectProperty("family.${module.id}.path", module.path)
            val pendingReason = modelFamilyTokenizerMetadataPendingReason(module.id)
            if (pendingReason.isNotBlank()) {
                expectProperty("family.${module.id}.tokenizerMetadataPendingReason", pendingReason)
            }
        }

        if (problems.isNotEmpty()) {
            throw GradleException(
                "Model-family bundle manifest validation failed:\n" +
                        problems.joinToString("\n") { "  - $it" }
            )
        }
    }
}

val validateModelFamilyBundlePresets = tasks.register("validateModelFamilyBundlePresets") {
    group = "verification"
    description = "Validate every production model-family bundle preset against its simulated selectors and policies."

    inputs.property("modelFamilyBundlePresets", modelFamilyBundlePresetInput())
    inputs.property(
        "modelFamilyProfiles",
        modelFamilyModules.joinToString(",") { "${it.id}:${it.profile}:${it.path}" }
    )
    inputs.property(
        "modelFamilyBundleAliases",
        modelFamilyBundleAliases.joinToString(",") { "${it.id}:${it.familyIds.sorted().joinToString("+")}" }
    )

    doLast {
        val problems = mutableListOf<String>()

        for (validation in modelFamilyBundlePresetValidations()) {
            val prefix = "${validation.preset.id} [selectors=${
                validation.preset.selectors.sorted().joinToString(",")
            }]"

            validation.configurationProblems.forEach { problem ->
                problems += "$prefix: $problem"
            }
            if (validation.missingRequiredFamilyIds.isNotEmpty()) {
                problems += "$prefix: missing required model-family plugins: ${
                    validation.missingRequiredFamilyIds.sorted().joinToString(", ")
                }"
            }
            if (validation.selectedForbiddenFamilyIds.isNotEmpty()) {
                problems += "$prefix: selected forbidden model-family plugins: ${
                    validation.selectedForbiddenFamilyIds.sorted().joinToString(", ")
                }"
            }
            for ((aliasId, missing) in validation.missingRequiredAliases) {
                problems += "$prefix: required model-family alias '$aliasId' is missing: ${
                    missing.sorted().joinToString(", ")
                }"
            }
            for ((aliasId, selectedFamilies) in validation.selectedForbiddenAliases) {
                problems += "$prefix: forbidden model-family alias '$aliasId' selected: ${
                    selectedFamilies.sorted().joinToString(", ")
                }"
            }
            if (!validation.productionSafetyPassed) {
                problems += "$prefix: production presets cannot select pending tokenizer metadata families: ${
                    validation.pendingTokenizerReasons.toSortedMap()
                        .entries
                        .joinToString(", ") { (familyId, reason) -> "$familyId ($reason)" }
                }"
            }
        }

        if (problems.isNotEmpty()) {
            throw GradleException(
                "Model-family bundle preset validation failed:\n" +
                        problems.joinToString("\n") { "  - $it" } +
                        "\nAdjust ModelFamilyBundlePreset selectors, policy fields, or alias membership " +
                        "and keep pending-tokenizer families out of prod_* presets before using these presets " +
                        "in production CI."
            )
        }
    }
}

val validateModelFamilyBundlePresetConformance = tasks.register("validateModelFamilyBundlePresetConformance") {
    group = "verification"
    description = "Validate the active model-family bundle still conforms to its reviewed preset."

    inputs.property("requestedModelFamilyBundlePreset", requestedModelFamilyBundlePreset?.id ?: "")
    inputs.property("requestedModelFamilyTokens", requestedModelFamilyTokens.sorted().joinToString(","))
    inputs.property("requestedModelFamilySelectorSource", requestedModelFamilySelectorSource)
    inputs.property("modelFamilyPolicySource", modelFamilyPolicySource)
    inputs.property("requiredModelFamilyTokens", requiredModelFamilyTokens.sorted().joinToString(","))
    inputs.property("forbiddenModelFamilyTokens", forbiddenModelFamilyTokens.sorted().joinToString(","))
    inputs.property("requiredModelFamilyAliasTokens", requiredModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("forbiddenModelFamilyAliasTokens", forbiddenModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("requireCleanModelFamilyBundlePresetConformance", requireCleanModelFamilyBundlePresetConformance)
    inputs.property("modelFamilyBundlePresets", modelFamilyBundlePresetInput())

    doLast {
        val conformance = modelFamilyBundlePresetConformanceValidation()
        if (!conformance.hasPreset) {
            println("No active model-family bundle preset; conformance validation skipped.")
            return@doLast
        }

        val problems = mutableListOf<String>()
        if (!conformance.presetMetadataPresent) {
            problems += "active preset '${conformance.presetId}' has no packaged preset metadata"
        }
        if (!conformance.selectorsMatch) {
            if (conformance.selectorAdditions.isNotEmpty()) {
                problems += "selectors add ${conformance.selectorAdditions.sorted().joinToString(", ")}"
            }
            if (conformance.selectorOmissions.isNotEmpty()) {
                problems += "selectors omit ${conformance.selectorOmissions.sorted().joinToString(", ")}"
            }
        }
        if (!conformance.policyInputsMatch) {
            if (conformance.requiredFamilyAdditions.isNotEmpty()) {
                problems += "required family policy adds ${
                    conformance.requiredFamilyAdditions.sorted().joinToString(", ")
                }"
            }
            if (conformance.requiredFamilyOmissions.isNotEmpty()) {
                problems += "required family policy omits ${
                    conformance.requiredFamilyOmissions.sorted().joinToString(", ")
                }"
            }
            if (conformance.forbiddenFamilyAdditions.isNotEmpty()) {
                problems += "forbidden family policy adds ${
                    conformance.forbiddenFamilyAdditions.sorted().joinToString(", ")
                }"
            }
            if (conformance.forbiddenFamilyOmissions.isNotEmpty()) {
                problems += "forbidden family policy omits ${
                    conformance.forbiddenFamilyOmissions.sorted().joinToString(", ")
                }"
            }
            if (conformance.requiredAliasAdditions.isNotEmpty()) {
                problems += "required alias policy adds ${
                    conformance.requiredAliasAdditions.sorted().joinToString(", ")
                }"
            }
            if (conformance.requiredAliasOmissions.isNotEmpty()) {
                problems += "required alias policy omits ${
                    conformance.requiredAliasOmissions.sorted().joinToString(", ")
                }"
            }
            if (conformance.forbiddenAliasAdditions.isNotEmpty()) {
                problems += "forbidden alias policy adds ${
                    conformance.forbiddenAliasAdditions.sorted().joinToString(", ")
                }"
            }
            if (conformance.forbiddenAliasOmissions.isNotEmpty()) {
                problems += "forbidden alias policy omits ${
                    conformance.forbiddenAliasOmissions.sorted().joinToString(", ")
                }"
            }
        }
        if (requireCleanModelFamilyBundlePresetConformance && !conformance.cleanPresetBuild) {
            if (conformance.explicitSelectorOverride) {
                problems += "explicit -Pgollek.modelFamilies override is not allowed in clean preset mode"
            }
            if (conformance.explicitPolicyOverride) {
                problems += "explicit model-family policy override is not allowed in clean preset mode"
            }
        }

        if (problems.isNotEmpty()) {
            throw GradleException(
                "Model-family bundle preset conformance failed for '${conformance.presetId}' " +
                        "(${conformance.compactSummary}):\n" +
                        problems.joinToString("\n") { "  - $it" } +
                        "\nUse the preset without selector/policy overrides, intentionally choose a different preset, " +
                        "or run without :ui:gollek-cli:validateModelFamilyBundlePresetConformance for ad hoc builds."
            )
        }

        println("Model-family bundle preset conformance: ${conformance.compactSummary}")
    }
}

val validateModelFamilyBundlePolicy = tasks.register("validateModelFamilyBundlePolicy") {
    group = "verification"
    description = "Validate required/forbidden model-family bundle policy for production builds."

    inputs.property("requestedModelFamilyTokens", requestedModelFamilyTokens.sorted().joinToString(","))
    inputs.property("requestedModelFamilyBundlePreset", requestedModelFamilyBundlePreset?.id ?: "")
    inputs.property("modelFamilyBundlePresets", modelFamilyBundlePresetInput())
    inputs.property("modelFamilyPolicySource", modelFamilyPolicySource)
    inputs.property("bundledModelFamilies", bundledModelFamilies.sorted().joinToString(","))
    inputs.property("requiredModelFamilies", requiredModelFamilies.sorted().joinToString(","))
    inputs.property("forbiddenModelFamilies", forbiddenModelFamilies.sorted().joinToString(","))
    inputs.property("requiredModelFamilyAliases", requiredModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property("forbiddenModelFamilyAliases", forbiddenModelFamilyAliasTokens.sorted().joinToString(","))
    inputs.property(
        "modelFamilyBundleAliases",
        modelFamilyBundleAliases.joinToString(",") { "${it.id}:${it.familyIds.sorted().joinToString("+")}" }
    )
    inputs.property("pendingModelFamilyTokenizerMetadata", pendingModelFamilyTokenizerMetadataInput())

    doLast {
        val missingRequired = modelFamilyPolicyMissingRequired()
        val selectedForbidden = modelFamilyPolicySelectedForbidden()
        val missingRequiredAliases = modelFamilyPolicyMissingRequiredAliases()
        val selectedForbiddenAliases = modelFamilyPolicySelectedForbiddenAliases()
        val selectedPendingTokenizerFamilies = bundledModelFamilies.intersect(pendingModelFamilyTokenizerMetadata)
        val problems = mutableListOf<String>()

        if (missingRequired.isNotEmpty()) {
            problems += "missing required model-family plugins: ${missingRequired.sorted().joinToString(", ")}"
        }
        if (selectedForbidden.isNotEmpty()) {
            problems += "selected forbidden model-family plugins: ${selectedForbidden.sorted().joinToString(", ")}"
        }
        for ((aliasId, missing) in missingRequiredAliases) {
            problems += "required model-family alias '$aliasId' is missing: ${missing.sorted().joinToString(", ")}"
        }
        for ((aliasId, selectedFamilies) in selectedForbiddenAliases) {
            problems += "forbidden model-family alias '$aliasId' selected: ${
                selectedFamilies.sorted().joinToString(", ")
            }"
        }
        if (requestedModelFamilyBundlePreset?.id?.startsWith("prod_") == true &&
            selectedPendingTokenizerFamilies.isNotEmpty()
        ) {
            problems += "production preset '${requestedModelFamilyBundlePreset.id}' selected pending tokenizer " +
                    "metadata families: ${
                        selectedPendingTokenizerFamilies.sorted()
                            .joinToString(", ") { familyId ->
                                "$familyId (${modelFamilyTokenizerMetadataPendingReason(familyId)})"
                            }
                    }"
        }

        if (problems.isNotEmpty()) {
            throw GradleException(
                "Model-family bundle policy validation failed for selectors " +
                        "'${requestedModelFamilyTokens.sorted().joinToString(",")}':\n" +
                        problems.joinToString("\n") { "  - $it" } +
                        "\nUse -Pgollek.modelFamilies=<families/profiles> to adjust the attached model farm, " +
                        "-Pgollek.requiredModelFamilies=<families/profiles> for required plugins, " +
                        "-Pgollek.forbiddenModelFamilies=<families/profiles> for forbidden plugins, " +
                        "-Pgollek.requiredModelFamilyAliases=<aliases> for complete alias coverage, " +
                        "-Pgollek.forbiddenModelFamilyAliases=<aliases> to reject alias overlap, " +
                        "or keep pending-tokenizer families out of prod_* presets."
            )
        }
    }
}

val writeModelFamilyBundleLock = tasks.register("writeModelFamilyBundleLock") {
    group = "help"
    description = "Write a checked-in model-family bundle lock for CI drift validation."
    dependsOn(validateModelFamilyBundlePolicy)
    dependsOn(validateModelFamilyFixtures)

    val outputFile = modelFamilyBundleLockFile
    outputs.file(outputFile)
    inputs.property("modelFamilyBundleLockPath", modelFamilyBundleLockPath)
    inputs.property("requestedModelFamilyBundlePreset", requestedModelFamilyBundlePreset?.id ?: "")
    inputs.property("modelFamilyBundlePresets", modelFamilyBundlePresetInput())
    inputs.property("currentModelFamilyBundleLock", modelFamilyBundleLockInput(selectedModelFamilyModules()))

    doLast {
        val selected = selectedModelFamilyModules()
        val file = outputFile.asFile
        file.parentFile.mkdirs()
        file.writeText(modelFamilyBundleLockText(selected), Charsets.UTF_8)
        println("Wrote model-family bundle lock: ${file.relativeTo(projectDir)}")
    }
}

val writeModelFamilyBundleLockDriftReport = tasks.register("writeModelFamilyBundleLockDriftReport") {
    group = "verification"
    description = "Write a JSON model-family bundle lock drift report for CI artifacts."
    mustRunAfter(writeModelFamilyBundleLock)

    val inputFile = modelFamilyBundleLockFile
    val outputFile = modelFamilyBundleLockDriftReportFile
    inputs.file(inputFile)
    outputs.file(outputFile)
    inputs.property("modelFamilyBundleLockPath", modelFamilyBundleLockPath)
    inputs.property("requestedModelFamilyBundlePreset", requestedModelFamilyBundlePreset?.id ?: "")
    inputs.property("modelFamilyBundlePresets", modelFamilyBundlePresetInput())
    inputs.property("currentModelFamilyBundleLock", modelFamilyBundleLockInput(selectedModelFamilyModules()))

    doLast {
        val lockFile = inputFile.asFile
        val actual = if (lockFile.isFile) {
            Properties().also { properties ->
                lockFile.inputStream().use(properties::load)
            }
        } else {
            null
        }
        val file = outputFile.get().asFile
        file.parentFile.mkdirs()
        file.writeText(
            modelFamilyBundleLockDriftReportJson(
                lockFile,
                modelFamilyBundleLockEntries(selectedModelFamilyModules()),
                actual,
                missingLock = !lockFile.isFile
            ),
            Charsets.UTF_8
        )
        println("Wrote model-family bundle lock drift report: ${file.relativeTo(projectDir)}")
    }
}

tasks.register("printModelFamilyBundleLockDriftJson") {
    group = "verification"
    description = "Print model-family bundle lock drift as JSON for CI/release automation."
    dependsOn(writeModelFamilyBundleLockDriftReport)
    doLast {
        println(modelFamilyBundleLockDriftReportFile.get().asFile.readText(Charsets.UTF_8))
    }
}

val validateModelFamilyBundleLockDriftReportContract = tasks.register("validateModelFamilyBundleLockDriftReportContract") {
    group = "verification"
    description = "Validate the JSON contract emitted by model-family bundle lock drift reports."

    inputs.property("modelFamilyBundleLockDriftReportContractSchema", 1)

    doLast {
        val expected = linkedMapOf(
            "familyCount" to "5",
            "families" to "gemma,llama,mistral,phi,qwen",
            "productionSafetyPassed" to "true",
            "pendingTokenizerFamilies" to "kimi",
            "pendingTokenizerReasons" to "kimi=tokenizer adapter pending descriptor stabilization",
            "fixtureRequiredContentFingerprints" to "gemma=abc,llama=def",
            "presetConformanceStatus" to "clean",
            "fixturePassed" to "true"
        )
        val actual = Properties().apply {
            setProperty("familyCount", "4")
            setProperty("families", "gemma,llama,mistral,qwen")
            setProperty("productionSafetyPassed", "false")
            setProperty("pendingTokenizerFamilies", "")
            setProperty("pendingTokenizerReasons", "")
            setProperty("fixtureRequiredContentFingerprints", "gemma=abc,llama=old")
            setProperty("presetConformanceStatus", "clean")
            setProperty("fixturePassed", "true")
        }
        val report = modelFamilyBundleLockDriftReportJson(
            File(projectDir, "model-family-bundle.contract.lock.properties"),
            expected,
            actual,
            missingLock = false
        )
        val requiredSnippets = linkedMapOf(
            "failed drift status" to "\"passed\": false",
            "raw drift count" to "\"driftCount\": 6",
            "scalar drift count" to "\"bundleScalarDriftCount\": 3",
            "selection drift count" to "\"bundleSelectionDriftCount\": 2",
            "production safety drift count" to "\"productionSafetyDriftCount\": 3",
            "fixture fingerprint drift count" to "\"fixtureFingerprintDriftCount\": 1",
            "scalar drift entry" to "{\"key\": \"familyCount\", " +
                    "\"scope\": \"bundle\", \"label\": \"family count\", \"status\": \"changed\", " +
                    "\"expected\": \"5\", \"actual\": \"4\"}",
            "selection drift entry" to "{\"key\": \"families\", " +
                    "\"scope\": \"family\", \"value\": \"phi\", \"status\": \"added\"}",
            "production safety summary" to "\"productionSafetySummaries\": [",
            "production safety drift entry" to "\"productionSafetyDrift\": [",
            "fixture fingerprint drift entry" to "\"fixtureFingerprintDrift\": " +
                    "[{\"key\": \"fixtureRequiredContentFingerprints\", \"scope\": \"required\", " +
                    "\"familyId\": \"llama\", \"status\": \"changed\", \"expected\": \"def\", " +
                    "\"actual\": \"old\"}]",
            "current bundle status" to "\"currentBundleStatus\": {",
            "current production safety status" to "\"productionSafety\": {",
            "current production pending tokenizer reasons" to "\"pendingTokenizerReasons\": {",
            "raw drift array" to "\"drift\": ["
        )
        val missing = requiredSnippets.entries
            .filterNot { (_, snippet) -> report.contains(snippet) }
            .map { it.key }
        if (missing.isNotEmpty()) {
            throw GradleException(
                "Model-family bundle lock drift report contract changed unexpectedly:\n" +
                        missing.joinToString("\n") { "  - missing $it" } +
                        "\nUpdate validateModelFamilyBundleLockDriftReportContract if this JSON change is intentional."
            )
        }
    }
}

val validateModelFamilyModuleCatalogReportContract =
    tasks.register("validateModelFamilyModuleCatalogReportContract") {
        group = "verification"
        description = "Validate the JSON contract emitted by model-family module catalog reports."

        inputs.property("modelFamilyModuleCatalogReportContractSchema", 1)
        inputs.property("catalogModelFamilyProjectPaths", catalogModelFamilyProjectPaths().sorted().joinToString(","))
        inputs.property("includedModelFamilyProjectPaths", includedModelFamilyProjectPaths().sorted().joinToString(","))
        inputs.property("pendingModelFamilyTokenizerMetadata", pendingModelFamilyTokenizerMetadataInput())

        doLast {
            val report = modelFamilyModuleCatalogReportJson()
            val summary = jsonModelFamilyModuleCatalogSummary()
            val requiredReportSnippets = linkedMapOf(
                "schema version" to "\"schemaVersion\": 1",
                "pass status" to "\"passed\":",
                "catalog family count" to "\"catalogFamilyCount\":",
                "catalog project count" to "\"catalogProjectCount\":",
                "included model project count" to "\"includedModelProjectCount\":",
                "included cataloged project count" to "\"includedCatalogedProjectCount\":",
                "support-only project count" to "\"supportOnlyProjectCount\":",
                "service descriptor complete project count" to "\"serviceDescriptorCompleteProjectCount\":",
                "service descriptor incomplete project count" to "\"serviceDescriptorIncompleteProjectCount\":",
                "tokenizer metadata ready family count" to "\"tokenizerMetadataReadyFamilyCount\":",
                "tokenizer metadata pending family count" to "\"tokenizerMetadataPendingFamilyCount\":",
                "tokenizer metadata missing family count" to "\"tokenizerMetadataMissingFamilyCount\":",
                "production ready family count" to "\"productionReadyFamilyCount\":",
                "direct safetensor ready family count" to "\"directSafetensorReadyFamilyCount\":",
                "production readiness pending family count" to "\"productionReadinessPendingFamilyCount\":",
                "direct safetensor pending family count" to "\"directSafetensorPendingFamilyCount\":",
                "metadata-only family count" to "\"metadataOnlyFamilyCount\":",
                "selected family count" to "\"selectedFamilyCount\":",
                "missing project count" to "\"missingProjectCount\":",
                "uncataloged project count" to "\"uncatalogedProjectCount\":",
                "readiness counts" to "\"readinessCounts\":",
                "selected families" to "\"selectedFamilies\":",
                "production ready families" to "\"productionReadyFamilies\":",
                "direct safetensor ready families" to "\"directSafetensorReadyFamilies\":",
                "production readiness pending families" to "\"productionReadinessPendingFamilies\":",
                "direct safetensor pending families" to "\"directSafetensorPendingFamilies\":",
                "metadata-only families" to "\"metadataOnlyFamilies\":",
                "missing projects" to "\"missingProjects\":",
                "uncataloged projects" to "\"uncatalogedProjects\":",
                "support-only projects" to "\"supportOnlyProjects\":",
                "service descriptor incomplete projects" to "\"serviceDescriptorIncompleteProjects\":",
                "tokenizer metadata pending families" to "\"tokenizerMetadataPendingFamilies\":",
                "tokenizer metadata pending reasons" to "\"tokenizerMetadataPendingReasons\":",
                "tokenizer metadata missing families" to "\"tokenizerMetadataMissingFamilies\":",
                "model family service descriptor presence" to "\"modelFamilyServiceDescriptorPresent\":",
                "gollek plugin service descriptor presence" to "\"gollekPluginServiceDescriptorPresent\":",
                "service descriptor completeness" to "\"serviceDescriptorComplete\":",
                "tokenizer metadata presence" to "\"tokenizerMetadataPresent\":",
                "tokenizer metadata pending flag" to "\"tokenizerMetadataPending\":",
                "tokenizer metadata pending reason" to "\"tokenizerMetadataPendingReason\":",
                "tokenizer kinds" to "\"tokenizerKinds\":",
                "tokenizer ready flag" to "\"tokenizerReady\":",
                "metadata-only flag" to "\"metadataOnly\":",
                "direct safetensor ready flag" to "\"directSafetensorReady\":",
                "production ready flag" to "\"productionReady\":",
                "readiness state" to "\"readiness\":",
                "catalog array" to "\"catalog\": [",
                "included projects array" to "\"includedProjects\": ["
            )
            val requiredSummarySnippets = linkedMapOf(
                "summary pass status" to "\"passed\":",
                "summary catalog family count" to "\"catalogFamilyCount\":",
                "summary production ready family count" to "\"productionReadyFamilyCount\":",
                "summary direct safetensor ready family count" to "\"directSafetensorReadyFamilyCount\":",
                "summary production readiness pending family count" to "\"productionReadinessPendingFamilyCount\":",
                "summary direct safetensor pending family count" to "\"directSafetensorPendingFamilyCount\":",
                "summary metadata-only family count" to "\"metadataOnlyFamilyCount\":",
                "summary readiness counts" to "\"readinessCounts\":",
                "summary production ready families" to "\"productionReadyFamilies\":",
                "summary direct safetensor ready families" to "\"directSafetensorReadyFamilies\":",
                "summary production readiness pending families" to "\"productionReadinessPendingFamilies\":",
                "summary direct safetensor pending families" to "\"directSafetensorPendingFamilies\":",
                "summary metadata-only families" to "\"metadataOnlyFamilies\":",
                "summary missing project count" to "\"missingProjectCount\":",
                "summary uncataloged project count" to "\"uncatalogedProjectCount\":",
                "summary support-only projects" to "\"supportOnlyProjects\":",
                "summary service descriptor incomplete projects" to "\"serviceDescriptorIncompleteProjects\":",
                "summary tokenizer metadata pending families" to "\"tokenizerMetadataPendingFamilies\":",
                "summary tokenizer metadata pending reasons" to "\"tokenizerMetadataPendingReasons\":",
                "summary tokenizer metadata missing families" to "\"tokenizerMetadataMissingFamilies\":"
            )
            val missing = requiredReportSnippets.entries
                .filterNot { (_, snippet) -> report.contains(snippet) }
                .map { it.key } +
                    requiredSummarySnippets.entries
                        .filterNot { (_, snippet) -> summary.contains(snippet) }
                        .map { it.key }
            if (missing.isNotEmpty()) {
                throw GradleException(
                    "Model-family module catalog report contract changed unexpectedly:\n" +
                            missing.joinToString("\n") { "  - missing $it" } +
                            "\nUpdate validateModelFamilyModuleCatalogReportContract if this JSON change is intentional."
                )
            }
        }
    }

val validateModelFamilyBundleLock = tasks.register("validateModelFamilyBundleLock") {
    group = "verification"
    description = "Validate the current model-family bundle against a checked-in lock."
    mustRunAfter(writeModelFamilyBundleLock)
    dependsOn(writeModelFamilyBundleLockDriftReport)

    val inputFile = modelFamilyBundleLockFile
    inputs.file(inputFile)
    inputs.property("modelFamilyBundleLockPath", modelFamilyBundleLockPath)
    inputs.property("requestedModelFamilyBundlePreset", requestedModelFamilyBundlePreset?.id ?: "")
    inputs.property("modelFamilyBundlePresets", modelFamilyBundlePresetInput())
    inputs.property("currentModelFamilyBundleLock", modelFamilyBundleLockInput(selectedModelFamilyModules()))

    doLast {
        val file = inputFile.asFile
        if (!file.isFile) {
            throw GradleException(
                "Model-family bundle lock validation failed: missing ${file.relativeTo(projectDir)}. " +
                        "Run :ui:gollek-cli:writeModelFamilyBundleLock with the intended " +
                        "-Pgollek.modelFamilies and policy selectors, then commit the lock. " +
                        "Report: ${modelFamilyBundleLockDriftReportFile.get().asFile.relativeTo(projectDir)}"
            )
        }

        val expected = modelFamilyBundleLockEntries(selectedModelFamilyModules())
        val actual = Properties()
        file.inputStream().use(actual::load)
        val driftEntries = modelFamilyBundleLockDriftEntries(expected, actual)
        val problems = modelFamilyBundleLockKeyProblems(driftEntries)

        if (problems.isNotEmpty()) {
            val conformanceSummaries = modelFamilyBundleLockDriftSummaries(expected, actual)
            val selectionSummaries = modelFamilyBundleLockSelectionDriftSummaries(expected, actual)
            val fixtureSummaries = modelFamilyBundleLockFixtureDriftSummaries(expected, actual)
            throw GradleException(
                "Model-family bundle lock drift detected in ${file.relativeTo(projectDir)}:\n" +
                        (
                                if (conformanceSummaries.isEmpty()) {
                                    ""
                                } else {
                                    "Preset conformance drift:\n" +
                                            conformanceSummaries.joinToString("\n") { "  - $it" } +
                                            "\n"
                                }
                                ) +
                        (
                                if (selectionSummaries.isEmpty()) {
                                    ""
                                } else {
                                    "Bundle selection drift:\n" +
                                            selectionSummaries.joinToString("\n") { "  - $it" } +
                                            "\n"
                                }
                                ) +
                        (
                                if (fixtureSummaries.isEmpty()) {
                                    ""
                                } else {
                                    "Fixture coverage drift:\n" +
                                            fixtureSummaries.joinToString("\n") { "  - $it" } +
                                            "\n"
                                }
                                ) +
                        "Raw lock key drift:\n" +
                        problems.joinToString("\n") { "  - $it" } +
                        "\nReport: ${modelFamilyBundleLockDriftReportFile.get().asFile.relativeTo(projectDir)}" +
                        "\nIf this change is intentional, rerun :ui:gollek-cli:writeModelFamilyBundleLock " +
                        "with the same production selectors and review the diff."
            )
        }
    }
}

tasks.named("compileJava") {
    dependsOn(validateModelFamilyModuleCatalog)
    dependsOn(validateModelFamilyBundle)
    dependsOn(validateExtensionAvailabilityProviders)
    dependsOn(validateModelFamilyBundleManifest)
    dependsOn(validateModelFamilyBundlePresets)
    dependsOn(validateModelFamilyBundlePolicy)
}

tasks.named("check") {
    dependsOn(validateModelFamilyModuleCatalog)
    dependsOn(validateModelFamilyBundle)
    dependsOn(validateExtensionAvailabilityProviders)
    dependsOn(validateExtensionAvailabilityProviderContracts)
    dependsOn(writeExtensionAvailabilityGateReport)
    dependsOn(validateModelFamilyBundleGate)
    dependsOn(writeModelFamilyBundleGateReport)
    dependsOn(validatePluginGates)
    dependsOn(writePluginGatesReport)
    dependsOn(validateModelFamilyFixtures)
    dependsOn(validateModelFamilyBundleManifest)
    dependsOn(validateModelFamilyBundlePresets)
    dependsOn(validateModelFamilyBundlePolicy)
    dependsOn(validateModelFamilyBundleLockDriftReportContract)
    dependsOn(validateModelFamilyModuleCatalogReportContract)
    if (enforceModelFamilyBundleLock) {
        dependsOn(validateModelFamilyBundleLock)
    }
    if (enforceModelFamilyFixtureFingerprintLock) {
        dependsOn(validateModelFamilyFixtureFingerprintLock)
    }
    if (enforceModelFamilyBundlePresetConformance) {
        dependsOn(validateModelFamilyBundlePresetConformance)
    }
}

tasks.test {
    useJUnitPlatform()
}
