// ArchitectureMapper.java
package tech.kayys.gollek.spi.model.mapper;

import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Maps GGUF architecture strings to HuggingFace class names.
 * This extracts the 800+ line mapping from the original god class.
 */
public class ArchitectureMapper {
    
    /**
     * Maps GGUF architecture string to HuggingFace class name.
     */
    public String mapGgufToHfClassName(String arch) {
        String normalized = arch == null ? "" : arch.toLowerCase(Locale.ROOT);
        
        // Gemma family
        if (normalized.contains("shieldgemma2") || normalized.contains("shield_gemma2")
                || normalized.contains("shield-gemma2")) {
            return "ShieldGemma2ForImageClassification";
        }
        if (normalized.contains("t5gemma2") || normalized.contains("t5_gemma2")
                || normalized.contains("t5-gemma2")) {
            return "T5Gemma2ForConditionalGeneration";
        }
        if (normalized.contains("t5gemma") || normalized.contains("t5_gemma")
                || normalized.contains("t5-gemma")) {
            return "T5GemmaForConditionalGeneration";
        }
        if (normalized.contains("vaultgemma") || normalized.contains("vault_gemma")
                || normalized.contains("vault-gemma")) {
            return "VaultGemmaForCausalLM";
        }
        if (normalized.contains("gemma4_unified") || normalized.contains("gemma4-unified")
                || normalized.contains("gemma-4-unified")) {
            return "Gemma4ForMultimodalLM";
        }
        if (normalized.contains("gemma4_audio") || normalized.contains("gemma4-audio")
                || normalized.contains("gemma-4-audio")) {
            return "Gemma4AudioModel";
        }
        if (normalized.contains("gemma4_vision") || normalized.contains("gemma4-vision")
                || normalized.contains("gemma-4-vision")) {
            return "Gemma4VisionModel";
        }
        if (normalized.contains("gemma4_text") || normalized.contains("gemma4-text")
                || normalized.contains("gemma-4-text")) {
            return "Gemma4ForCausalLM";
        }
        if (normalized.contains("gemma4") || normalized.contains("gemma-4")) {
            return "Gemma4ForConditionalGeneration";
        }
        if (normalized.contains("gemma3n")) {
            return "Gemma3nForConditionalGeneration";
        }
        if (normalized.contains("paligemma") || normalized.contains("pali_gemma")) {
            return "PaliGemmaForConditionalGeneration";
        }
        if (normalized.contains("recurrent_gemma") || normalized.contains("recurrent-gemma")) {
            return "RecurrentGemmaForCausalLM";
        }
        if (normalized.contains("gemma3")) {
            return "Gemma3ForCausalLM";
        }
        if (normalized.contains("gemma2")) {
            return "Gemma2ForCausalLM";
        }
        if (normalized.contains("gemma")) {
            return "GemmaForCausalLM";
        }
        
        // Qwen family
        if (normalized.contains("colqwen2")) {
            return "ColQwen2ForRetrieval";
        }
        if (normalized.contains("qwen3_omni_moe") || normalized.contains("qwen3-omni-moe")) {
            return "Qwen3OmniMoeForConditionalGeneration";
        }
        if (normalized.contains("qwen3_5_moe") || normalized.contains("qwen3.5_moe")
                || normalized.contains("qwen3-5-moe") || normalized.contains("qwen3.5-moe")) {
            return "Qwen3_5MoeForConditionalGeneration";
        }
        if (normalized.contains("qwen3_5") || normalized.contains("qwen3.5")
                || normalized.contains("qwen3-5")) {
            return "Qwen3_5ForConditionalGeneration";
        }
        if (normalized.contains("qwen3_vl_moe") || normalized.contains("qwen3-vl-moe")) {
            return "Qwen3VLMoeForConditionalGeneration";
        }
        if (normalized.contains("qwen3_vl") || normalized.contains("qwen3-vl")) {
            return "Qwen3VLForConditionalGeneration";
        }
        if (normalized.contains("qwen3_moe") || normalized.contains("qwen3-moe")) {
            return "Qwen3MoeForCausalLM";
        }
        if (normalized.contains("qwen3_next") || normalized.contains("qwen3-next")) {
            return "Qwen3NextForCausalLM";
        }
        if (normalized.contains("qwen3")) {
            return "Qwen3ForCausalLM";
        }
        if (normalized.contains("qwen2_5_omni") || normalized.contains("qwen2.5_omni")
                || normalized.contains("qwen2-5-omni") || normalized.contains("qwen2.5-omni")) {
            return "Qwen2_5OmniForConditionalGeneration";
        }
        if (normalized.contains("qwen2_5_vl") || normalized.contains("qwen2.5_vl")
                || normalized.contains("qwen2-5-vl") || normalized.contains("qwen2.5-vl")) {
            return "Qwen2_5_VLForConditionalGeneration";
        }
        if (normalized.contains("qwen2_vl") || normalized.contains("qwen2-vl")) {
            return "Qwen2VLForConditionalGeneration";
        }
        if (normalized.contains("qwen2_audio") || normalized.contains("qwen2-audio")) {
            return "Qwen2AudioForConditionalGeneration";
        }
        if (normalized.contains("qwen2_moe") || normalized.contains("qwen2-moe")) {
            return "Qwen2MoeForCausalLM";
        }
        if (normalized.contains("qwen")) {
            return "Qwen2ForCausalLM";
        }
        
        // Mistral family
        if (normalized.contains("ministral3") || normalized.contains("ministral_3")
                || normalized.contains("ministral-3")) {
            return "Ministral3ForCausalLM";
        }
        if (normalized.contains("ministral")) {
            return "MinistralForCausalLM";
        }
        if (normalized.contains("mixtral")) {
            return "MixtralForCausalLM";
        }
        if (normalized.contains("mistral4") || normalized.contains("mistral_4")
                || normalized.contains("mistral-4")) {
            return "Mistral4ForCausalLM";
        }
        if (normalized.contains("mistral3") || normalized.contains("mistral_3")
                || normalized.contains("mistral-3")) {
            return "Mistral3ForConditionalGeneration";
        }
        if (normalized.contains("mistral")) {
            return "MistralForCausalLM";
        }
        
        // Phi family
        if (normalized.contains("phi3")) {
            return "Phi3ForCausalLM";
        }
        if (normalized.contains("phi4_multimodal") || normalized.contains("phi4-multimodal")
                || normalized.contains("phi_4_multimodal") || normalized.contains("phi-4-multimodal")) {
            return "Phi4MultimodalForCausalLM";
        }
        if (normalized.contains("phimoe") || normalized.contains("phi_moe")
                || normalized.contains("phi-moe")) {
            return "PhimoeForCausalLM";
        }
        if (normalized.contains("phi")) {
            return "PhiForCausalLM";
        }
        
        // Falcon family
        if (normalized.contains("falcon_h1") || normalized.contains("falcon-h1")) {
            return "FalconH1ForCausalLM";
        }
        if (normalized.contains("falcon_mamba") || normalized.contains("falcon-mamba")) {
            return "FalconMambaForCausalLM";
        }
        if (normalized.contains("falcon")) {
            return "FalconForCausalLM";
        }
        
        // Llama family
        if (normalized.contains("llama4") || normalized.contains("llama-4")) {
            return "Llama4ForConditionalGeneration";
        }
        if (normalized.contains("mllama")) {
            return "MllamaForConditionalGeneration";
        }
        if (normalized.contains("diffllama") || normalized.contains("diff_llama")
                || normalized.contains("diff-llama")) {
            return "DiffLlamaForCausalLM";
        }
        if (normalized.contains("code_llama") || normalized.contains("code-llama")
                || normalized.contains("codellama")) {
            return "LlamaForCausalLM";
        }
        if (normalized.contains("llama")) {
            return "LlamaForCausalLM";
        }
        
        // DeepSeek family
        if (normalized.contains("deepseek_vl_hybrid") || normalized.contains("deepseek-vl-hybrid")) {
            return "DeepseekVLHybridForConditionalGeneration";
        }
        if (normalized.contains("deepseek_vl") || normalized.contains("deepseek-vl")) {
            return "DeepseekVLForConditionalGeneration";
        }
        if (normalized.contains("deepseek_v2") || normalized.contains("deepseek-v2")) {
            return "DeepseekV2ForCausalLM";
        }
        if (normalized.contains("deepseek_v3") || normalized.contains("deepseek-v3")
                || normalized.contains("deepseek_moe") || normalized.contains("deepseek-moe")
                || normalized.contains("deepseek_r1") || normalized.contains("deepseek-r1")
                || normalized.contains("deepseek")) {
            return "DeepseekV3ForCausalLM";
        }
        
        // Other popular models
        if (normalized.contains("cohere")) {
            return "Cohere2ForCausalLM";
        }
        if (normalized.contains("kimi")) {
            return "KimiForCausalLM";
        }
        if (normalized.equals("yi") || normalized.contains("yi_")) {
            return "YiForCausalLM";
        }
        if (normalized.contains("bloom")) {
            return "BloomForCausalLM";
        }
        if (normalized.contains("bitnet")) {
            return "BitNetForCausalLM";
        }
        if (normalized.contains("dbrx")) {
            return "DbrxForCausalLM";
        }
        if (normalized.contains("exaone_moe") || normalized.contains("exaone-moe")) {
            return "ExaoneMoeForCausalLM";
        }
        if (normalized.contains("exaone4")) {
            return "Exaone4ForCausalLM";
        }
        if (normalized.contains("bamba")) {
            return "BambaForCausalLM";
        }
        if (normalized.contains("zamba2")) {
            return "Zamba2ForCausalLM";
        }
        if (normalized.contains("zamba")) {
            return "ZambaForCausalLM";
        }
        if (normalized.contains("arcee")) {
            return "ArceeForCausalLM";
        }
        
        // Granite family
        if (normalized.contains("granitemoehybrid")) {
            return "GraniteMoeHybridForCausalLM";
        }
        if (normalized.contains("granitemoeshared")) {
            return "GraniteMoeSharedForCausalLM";
        }
        if (normalized.contains("granitemoe")) {
            return "GraniteMoeForCausalLM";
        }
        if (normalized.contains("granite")) {
            return "GraniteForCausalLM";
        }
        
        if (normalized.contains("nemotron")) {
            return "NemotronForCausalLM";
        }
        if (normalized.contains("stablelm") || normalized.contains("stable_lm")) {
            return "StableLmForCausalLM";
        }
        if (normalized.contains("persimmon")) {
            return "PersimmonForCausalLM";
        }
        if (normalized.contains("smollm3") || normalized.contains("smol_lm3")
                || normalized.contains("smol-lm3")) {
            return "SmolLM3ForCausalLM";
        }
        if (normalized.contains("xglm")) {
            return "XGLMForCausalLM";
        }
        
        // GLM family
        if (normalized.contains("glm_ocr_vision") || normalized.contains("glm-ocr-vision")) {
            return "GlmOcrVisionModel";
        }
        if (normalized.contains("glm_ocr_text") || normalized.contains("glm-ocr-text")) {
            return "GlmOcrTextModel";
        }
        if (normalized.contains("glm_ocr") || normalized.contains("glm-ocr")) {
            return "GlmOcrForConditionalGeneration";
        }
        if (normalized.equals("glm") || normalized.contains("glm_") || normalized.contains("glm-")) {
            return "GlmForCausalLM";
        }
        
        // Code models
        if (normalized.contains("starcoder2")) {
            return "Starcoder2ForCausalLM";
        }
        if (normalized.contains("gpt_bigcode")) {
            return "GPTBigCodeForCausalLM";
        }
        if (normalized.contains("gpt_oss") || normalized.contains("gpt-oss")) {
            return "GptOssForCausalLM";
        }
        if (normalized.contains("codegen")) {
            return "CodeGenForCausalLM";
        }
        
        // GPT family
        if (normalized.contains("gpt_neox_japanese") || normalized.contains("gpt-neox-japanese")
                || normalized.contains("gptneox_japanese")) {
            return "GPTNeoXJapaneseForCausalLM";
        }
        if (normalized.contains("gpt_neox") || normalized.contains("gptneox")) {
            return "GPTNeoXForCausalLM";
        }
        if (normalized.contains("gptj") || normalized.contains("gpt_j")) {
            return "GPTJForCausalLM";
        }
        if (normalized.contains("gpt_neo") || normalized.contains("gpt-neo")
                || normalized.contains("gptneo")) {
            return "GPTNeoForCausalLM";
        }
        if (normalized.contains("openai_gpt") || normalized.contains("openai-gpt")) {
            return "OpenAIGPTLMHeadModel";
        }
        if (normalized.contains("dialogpt") || normalized.contains("megatron_gpt2")
                || normalized.contains("megatron-gpt2") || normalized.contains("gpt2")) {
            return "GPT2LMHeadModel";
        }
        if (normalized.contains("mpt")) {
            return "MptForCausalLM";
        }
        
        // Mamba family
        if (normalized.contains("mamba2")) {
            return "Mamba2ForCausalLM";
        }
        if (normalized.contains("mamba")) {
            return "MambaForCausalLM";
        }
        if (normalized.contains("rwkv")) {
            return "RwkvForCausalLM";
        }
        if (normalized.contains("jamba")) {
            return "JambaForCausalLM";
        }
        
        // Olmo family
        if (normalized.contains("olmo_hybrid") || normalized.contains("olmo-hybrid")) {
            return "OlmoHybridForCausalLM";
        }
        if (normalized.contains("olmoe") || normalized.contains("olmo_moe")
                || normalized.contains("olmo-moe")) {
            return "OlmoeForCausalLM";
        }
        if (normalized.contains("olmo3")) {
            return "Olmo3ForCausalLM";
        }
        if (normalized.contains("olmo2")) {
            return "Olmo2ForCausalLM";
        }
        if (normalized.contains("olmo")) {
            return "OlmoForCausalLM";
        }
        
        // T5/BART family
        if (normalized.contains("plbart")) {
            return "PLBartForConditionalGeneration";
        }
        if (normalized.contains("mbart")) {
            return "MBartForConditionalGeneration";
        }
        if (normalized.contains("barthez") || normalized.contains("bartpho")) {
            return "BartForConditionalGeneration";
        }
        if (normalized.contains("blenderbot_small") || normalized.contains("blenderbot-small")) {
            return "BlenderbotSmallForConditionalGeneration";
        }
        if (normalized.contains("blenderbot")) {
            return "BlenderbotForConditionalGeneration";
        }
        if (normalized.contains("bart")) {
            return "BartForConditionalGeneration";
        }
        if (normalized.contains("marian")) {
            return "MarianMTModel";
        }
        if (normalized.contains("fsmt")) {
            return "FSMTForConditionalGeneration";
        }
        if (normalized.contains("m2m_100") || normalized.contains("m2m100")) {
            return "M2M100ForConditionalGeneration";
        }
        if (normalized.contains("nllb-moe") || normalized.contains("nllb_moe")) {
            return "NllbMoeForConditionalGeneration";
        }
        if (normalized.contains("nllb")) {
            return "M2M100ForConditionalGeneration";
        }
        if (normalized.contains("bigbird_pegasus") || normalized.contains("bigbird-pegasus")) {
            return "BigBirdPegasusForConditionalGeneration";
        }
        if (normalized.contains("big_bird") || normalized.contains("bigbird")) {
            return "BigBirdModel";
        }
        if (normalized.contains("pegasus_x") || normalized.contains("pegasus-x")) {
            return "PegasusXForConditionalGeneration";
        }
        if (normalized.contains("pegasus")) {
            return "PegasusForConditionalGeneration";
        }
        if (normalized.contains("prophetnet") || normalized.contains("prophet_net")) {
            return "ProphetNetForConditionalGeneration";
        }
        if (normalized.contains("switch_transformers") || normalized.contains("switch-transformers")) {
            return "SwitchTransformersForConditionalGeneration";
        }
        if (normalized.contains("longt5")) {
            return "LongT5ForConditionalGeneration";
        }
        if (normalized.contains("umt5")) {
            return "UMT5ForConditionalGeneration";
        }
        if (normalized.contains("mt5")) {
            return "MT5ForConditionalGeneration";
        }
        if (normalized.contains("byt5") || normalized.contains("t5")) {
            return "T5ForConditionalGeneration";
        }
        
        // Vision models
        if (normalized.contains("pixtral")) {
            return "PixtralVisionModel";
        }
        if (normalized.contains("fuyu")) {
            return "FuyuForCausalLM";
        }
        if (normalized.contains("kosmos2_5") || normalized.contains("kosmos2.5")
                || normalized.contains("kosmos-2.5") || normalized.contains("kosmos_2_5")) {
            return "Kosmos2_5ForConditionalGeneration";
        }
        if (normalized.contains("kosmos-2") || normalized.contains("kosmos2")
                || normalized.contains("kosmos_2")) {
            return "Kosmos2ForConditionalGeneration";
        }
        if (normalized.contains("instructblipvideo")) {
            return "InstructBlipVideoForConditionalGeneration";
        }
        if (normalized.contains("instructblip")) {
            return "InstructBlipForConditionalGeneration";
        }
        if (normalized.contains("video_llava") || normalized.contains("video-llava")) {
            return "VideoLlavaForConditionalGeneration";
        }
        if (normalized.contains("vipllava") || normalized.contains("vip_llava")) {
            return "VipLlavaForConditionalGeneration";
        }
        if (normalized.contains("llava_next_video") || normalized.contains("llava-next-video")) {
            return "LlavaNextVideoForConditionalGeneration";
        }
        if (normalized.contains("llava_next") || normalized.contains("llava-next")) {
            return "LlavaNextForConditionalGeneration";
        }
        if (normalized.contains("llava_onevision") || normalized.contains("llava-onevision")) {
            return "LlavaOnevisionForConditionalGeneration";
        }
        if (normalized.contains("llava")) {
            return "LlavaForConditionalGeneration";
        }
        
        // CLIP and vision encoders
        if (normalized.contains("florence2") || normalized.contains("florence_vision")) {
            return "Florence2ForConditionalGeneration";
        }
        if (normalized.contains("bridgetower")) {
            return "BridgeTowerModel";
        }
        if (normalized.contains("align_vision_model") || normalized.contains("align-vision-model")) {
            return "AlignVisionModel";
        }
        if (normalized.contains("align_text_model") || normalized.contains("align-text-model")) {
            return "AlignTextModel";
        }
        if (normalized.equals("align") || normalized.contains("align_")
                || normalized.contains("align-")) {
            return "AlignModel";
        }
        if (normalized.contains("visual_bert") || normalized.contains("visual-bert")) {
            return "VisualBertModel";
        }
        if (normalized.contains("lxmert")) {
            return "LxmertModel";
        }
        if (normalized.contains("vilt")) {
            return "ViltModel";
        }
        if (normalized.contains("siglip2")) {
            return "Siglip2Model";
        }
        if (normalized.contains("siglip")) {
            return "SiglipModel";
        }
        if (normalized.contains("clip")) {
            return "CLIPModel";
        }
        
        // Layout models
        if (normalized.contains("layoutxlm")) {
            return "LayoutLMv2Model";
        }
        if (normalized.contains("layoutlmv3")) {
            return "LayoutLMv3Model";
        }
        if (normalized.contains("layoutlmv2")) {
            return "LayoutLMv2Model";
        }
        if (normalized.contains("layoutlm")) {
            return "LayoutLMModel";
        }
        if (normalized.contains("markuplm") || normalized.contains("markup_lm")) {
            return "MarkupLMModel";
        }
        if (normalized.equals("bros") || normalized.contains("bros_") || normalized.contains("bros-")) {
            return "BrosModel";
        }
        if (normalized.equals("lilt") || normalized.contains("lilt_") || normalized.contains("lilt-")) {
            return "LiltModel";
        }
        
        // OCR models
        if (normalized.contains("mgp_str") || normalized.contains("mgp-str")) {
            return "MgpstrForSceneTextRecognition";
        }
        if (normalized.contains("trocr")) {
            return "TrOCRForCausalLM";
        }
        if (normalized.contains("pix2struct")) {
            return "Pix2StructForConditionalGeneration";
        }
        if (normalized.contains("nougat")) {
            return "VisionEncoderDecoderModel";
        }
        if (normalized.contains("lighton_ocr") || normalized.contains("lighton-ocr")) {
            return "LightOnOcrForConditionalGeneration";
        }
        if (normalized.contains("got_ocr2") || normalized.contains("got-ocr2")) {
            return "GotOcr2ForConditionalGeneration";
        }
        
        // Speech models
        if (normalized.contains("speecht5") || normalized.contains("speech_t5")) {
            return "SpeechT5Model";
        }
        if (normalized.contains("bark")) {
            return "BarkModel";
        }
        if (normalized.contains("musicgen_decoder")) {
            return "MusicgenForCausalLM";
        }
        if (normalized.contains("musicgen")) {
            return "MusicgenForConditionalGeneration";
        }
        if (normalized.contains("seamless_m4t_v2") || normalized.contains("seamless-m4t-v2")) {
            return "SeamlessM4Tv2Model";
        }
        if (normalized.contains("seamless_m4t") || normalized.contains("seamless-m4t")) {
            return "SeamlessM4TModel";
        }
        if (normalized.contains("wav2vec2_conformer") || normalized.contains("wav2vec2-conformer")) {
            return "Wav2Vec2ConformerModel";
        }
        if (normalized.contains("wav2vec2")) {
            return "Wav2Vec2Model";
        }
        if (normalized.contains("hubert")) {
            return "HubertModel";
        }
        if (normalized.contains("wavlm")) {
            return "WavLMModel";
        }
        if (normalized.contains("encodec")) {
            return "EncodecModel";
        }
        if (normalized.contains("clap")) {
            return "ClapModel";
        }
        if (normalized.contains("whisper")) {
            return "WhisperForConditionalGeneration";
        }
        
        // RAG and DPR
        if (normalized.contains("rag")) {
            return "RagModel";
        }
        if (normalized.contains("dpr")) {
            return "DPRQuestionEncoder";
        }
        
        // Vision-language models
        if (normalized.contains("video_llama_3") || normalized.contains("video-llama-3")) {
            return "VideoLlama3ForConditionalGeneration";
        }
        if (normalized.contains("internvl")) {
            return "InternVLForConditionalGeneration";
        }
        if (normalized.contains("smolvlm") || normalized.contains("smol_vlm")) {
            return "SmolVLMForConditionalGeneration";
        }
        if (normalized.contains("fast_vlm") || normalized.contains("fast-vlm")) {
            return "FastVlmForConditionalGeneration";
        }
        if (normalized.contains("idefics3")) {
            return "Idefics3ForConditionalGeneration";
        }
        if (normalized.contains("idefics2")) {
            return "Idefics2ForConditionalGeneration";
        }
        if (normalized.contains("idefics")) {
            return "IdeficsForVisionText2Text";
        }
        if (normalized.contains("chameleon")) {
            return "ChameleonForConditionalGeneration";
        }
        
        // Retrieval models
        if (normalized.contains("colmodernvbert")) {
            return "ColModernVBertForRetrieval";
        }
        if (normalized.contains("colpali")) {
            return "ColPaliForRetrieval";
        }
        
        // BLIP family
        if (normalized.contains("blip_2") || normalized.contains("blip-2") || normalized.contains("blip2")) {
            return "Blip2ForConditionalGeneration";
        }
        if (normalized.contains("blip")) {
            return "BlipForConditionalGeneration";
        }
        
        // SAM models
        if (normalized.contains("sam_hq") || normalized.contains("sam-hq")) {
            return "SamHQModel";
        }
        if (normalized.contains("sam2")) {
            return "Sam2Model";
        }
        if (normalized.equals("sam") || normalized.contains("segment_anything")) {
            return "SamModel";
        }
        if (normalized.contains("clipseg")) {
            return "CLIPSegForImageSegmentation";
        }
        if (normalized.contains("metaclip_2") || normalized.contains("metaclip-2")) {
            return "MetaClip2Model";
        }
        
        // Group and multi-modal
        if (normalized.contains("vision_text_dual_encoder")
                || normalized.contains("vision-text-dual-encoder")) {
            return "VisionTextDualEncoderModel";
        }
        if (normalized.contains("x_clip") || normalized.contains("x-clip")
                || normalized.contains("xclip")) {
            return "XCLIPModel";
        }
        
        // Chinese CLIP
        if (normalized.contains("altclip_vision_model") || normalized.contains("altclip-vision-model")) {
            return "AltCLIPVisionModel";
        }
        if (normalized.contains("altclip_text_model") || normalized.contains("altclip-text-model")) {
            return "AltCLIPTextModel";
        }
        if (normalized.contains("altclip")) {
            return "AltCLIPModel";
        }
        if (normalized.contains("chinese_clip_vision_model")
                || normalized.contains("chinese-clip-vision-model")) {
            return "ChineseCLIPVisionModel";
        }
        if (normalized.contains("chinese_clip_text_model")
                || normalized.contains("chinese-clip-text-model")) {
            return "ChineseCLIPTextModel";
        }
        if (normalized.contains("chinese_clip") || normalized.contains("chinese-clip")) {
            return "ChineseCLIPModel";
        }
        
        // GroupViT
        if (normalized.contains("groupvit_vision_model")
                || normalized.contains("groupvit-vision-model")) {
            return "GroupViTVisionModel";
        }
        if (normalized.contains("groupvit_text_model") || normalized.contains("groupvit-text-model")) {
            return "GroupViTTextModel";
        }
        if (normalized.contains("groupvit")) {
            return "GroupViTModel";
        }
        
        // Object detection
        if (normalized.contains("owlv2") || normalized.contains("owl_v2")
                || normalized.contains("owl-v2")) {
            return "Owlv2ForObjectDetection";
        }
        if (normalized.contains("owlvit") || normalized.contains("owl_vit")
                || normalized.contains("owl-vit")) {
            return "OwlViTForObjectDetection";
        }
        if (normalized.contains("mm_grounding_dino") || normalized.contains("mm-grounding-dino")) {
            return "MMGroundingDinoForObjectDetection";
        }
        if (normalized.contains("grounding_dino") || normalized.contains("grounding-dino")) {
            return "GroundingDinoForObjectDetection";
        }
        if (normalized.contains("deformable_detr") || normalized.contains("deformable-detr")) {
            return "DeformableDetrModel";
        }
        if (normalized.contains("conditional_detr") || normalized.contains("conditional-detr")) {
            return "ConditionalDetrModel";
        }
        if (normalized.contains("rt_detr") || normalized.contains("rt-detr")) {
            return "RTDetrModel";
        }
        if (normalized.contains("detr")) {
            return "DetrModel";
        }
        if (normalized.contains("yolos")) {
            return "YolosForObjectDetection";
        }
        
        // Segmentation
        if (normalized.contains("mask2former") || normalized.contains("mask2-former")) {
            return "Mask2FormerForUniversalSegmentation";
        }
        if (normalized.contains("maskformer")) {
            return "MaskFormerForInstanceSegmentation";
        }
        if (normalized.contains("oneformer")) {
            return "OneFormerForUniversalSegmentation";
        }
        if (normalized.contains("upernet")) {
            return "UperNetForSemanticSegmentation";
        }
        
        // Depth estimation
        if (normalized.contains("depth_anything") || normalized.contains("depth-anything")) {
            return "DepthAnythingForDepthEstimation";
        }
        if (normalized.contains("depth_pro") || normalized.contains("depth-pro")) {
            return "DepthProForDepthEstimation";
        }
        if (normalized.contains("zoedepth") || normalized.contains("zoe_depth")) {
            return "ZoeDepthForDepthEstimation";
        }
        if (normalized.equals("dpt") || normalized.contains("dpt_") || normalized.contains("dpt-")) {
            return "DPTForDepthEstimation";
        }
        
        // Vision transformers
        if (normalized.contains("segformer")) {
            return "SegformerModel";
        }
        if (normalized.contains("beit")) {
            return "BeitModel";
        }
        if (normalized.contains("dinov2_with_registers")
                || normalized.contains("dinov2-with-registers")) {
            return "Dinov2WithRegistersModel";
        }
        if (normalized.contains("dinov2")) {
            return "Dinov2Model";
        }
        if (normalized.contains("deit")) {
            return "DeiTModel";
        }
        if (normalized.contains("focalnet")) {
            return "FocalNetModel";
        }
        if (normalized.equals("cvt") || normalized.contains("cvt_") || normalized.contains("cvt-")) {
            return "CvtModel";
        }
        if (normalized.contains("levit")) {
            return "LevitModel";
        }
        if (normalized.contains("mobilevitv2") || normalized.contains("mobilevit_v2")
                || normalized.contains("mobilevit-v2")) {
            return "MobileViTV2Model";
        }
        if (normalized.contains("mobilevit")) {
            return "MobileViTModel";
        }
        if (normalized.contains("poolformer")) {
            return "PoolFormerModel";
        }
        if (normalized.contains("pvt_v2") || normalized.contains("pvt-v2")) {
            return "PvtV2Model";
        }
        if (normalized.equals("pvt") || normalized.contains("pvt_") || normalized.contains("pvt-")) {
            return "PvtModel";
        }
        if (normalized.contains("vit")) {
            return "ViTModel";
        }
        if (normalized.contains("donut")) {
            return "DonutSwinModel";
        }
        if (normalized.contains("swinv2") || normalized.contains("swin_v2")) {
            return "Swinv2Model";
        }
        if (normalized.contains("swin")) {
            return "SwinModel";
        }
        
        // CNN models
        if (normalized.contains("convnextv2") || normalized.contains("convnext_v2")) {
            return "ConvNextV2Model";
        }
        if (normalized.contains("convnext")) {
            return "ConvNextModel";
        }
        if (normalized.contains("efficientnet")) {
            return "EfficientNetModel";
        }
        if (normalized.contains("mobilenet_v2") || normalized.contains("mobilenet-v2")) {
            return "MobileNetV2Model";
        }
        if (normalized.contains("mobilenet_v1") || normalized.contains("mobilenet-v1")) {
            return "MobileNetV1Model";
        }
        if (normalized.contains("resnet")) {
            return "ResNetModel";
        }
        if (normalized.contains("regnet")) {
            return "RegNetModel";
        }
        if (normalized.contains("vits")) {
            return "VitsModel";
        }
        
        // Data2Vec
        if (normalized.contains("data2vec_audio") || normalized.contains("data2vec-audio")) {
            return "Data2VecAudioModel";
        }
        if (normalized.contains("data2vec_vision") || normalized.contains("data2vec-vision")) {
            return "Data2VecVisionModel";
        }
        if (normalized.contains("data2vec_text") || normalized.contains("data2vec-text")) {
            return "Data2VecTextModel";
        }
        
        // Long context
        if (normalized.equals("led") || normalized.contains("led_") || normalized.contains("led-")) {
            return "LEDForConditionalGeneration";
        }
        if (normalized.contains("longformer")) {
            return "LongformerModel";
        }
        if (normalized.contains("reformer")) {
            return "ReformerModelWithLMHead";
        }
        
        // BERT variants
        if (normalized.contains("modernbert_decoder")
                || normalized.contains("modernbert-decoder")
                || normalized.contains("modern_bert_decoder")) {
            return "ModernBertDecoderForCausalLM";
        }
        if (normalized.contains("modernbert") || normalized.contains("modern_bert")) {
            return "ModernBertModel";
        }
        if (normalized.contains("modernvbert") || normalized.contains("modern_vbert")
                || normalized.contains("modern-vbert")) {
            return "ModernVBertForMaskedLM";
        }
        if (normalized.contains("nomic_bert") || normalized.contains("nomic-bert")) {
            return "NomicBertModel";
        }
        if (normalized.contains("eurobert") || normalized.contains("euro_bert")) {
            return "EuroBertModel";
        }
        if (normalized.contains("distilbert") || normalized.contains("distil_bert")) {
            return "DistilBertModel";
        }
        if (normalized.contains("rembert") || normalized.contains("rem_bert")) {
            return "RemBertModel";
        }
        if (normalized.contains("mobilebert") || normalized.contains("mobile_bert")) {
            return "MobileBertModel";
        }
        if (normalized.contains("megatron_bert") || normalized.contains("megatron-bert")) {
            return "MegatronBertModel";
        }
        if (normalized.contains("squeezebert") || normalized.contains("squeeze_bert")) {
            return "SqueezeBertModel";
        }
        if (normalized.contains("bert_japanese") || normalized.contains("bert-japanese")) {
            return "BertModel";
        }
        if (normalized.contains("bert_generation") || normalized.contains("bert-generation")) {
            return "BertGenerationDecoder";
        }
        if (normalized.contains("convbert") || normalized.contains("conv_bert")) {
            return "ConvBertModel";
        }
        if (normalized.contains("roc_bert") || normalized.contains("roc-bert")) {
            return "RoCBertForMaskedLM";
        }
        if (normalized.contains("wav2vec2_bert") || normalized.contains("wav2vec2-bert")) {
            return "Wav2Vec2BertForCTC";
        }
        if (normalized.contains("bert")) {
            return "BertModel";
        }
        
        // RoBERTa variants
        if (normalized.contains("xlm_roberta_xl") || normalized.contains("xlm-roberta-xl")) {
            return "XLMRobertaXLModel";
        }
        if (normalized.contains("xlm_roberta") || normalized.contains("xlm-roberta")) {
            return "XLMRobertaModel";
        }
        if (normalized.contains("jina_embeddings_v3") || normalized.contains("jina-embeddings-v3")) {
            return "JinaEmbeddingsV3Model";
        }
        if (normalized.contains("camembert") || normalized.contains("bertweet")
                || normalized.contains("herbert") || normalized.contains("phobert")) {
            return "RobertaModel";
        }
        if (normalized.contains("roberta_prelayernorm")
                || normalized.contains("roberta-prelayernorm")
                || normalized.contains("roberta_pre_layer_norm")
                || normalized.contains("roberta-pre-layer-norm")) {
            return "RobertaPreLayerNormModel";
        }
        if (normalized.contains("roberta")) {
            return "RobertaModel";
        }
        
        // Other models
        if (normalized.contains("luke")) {
            return "LukeModel";
        }
        if (normalized.contains("funnel")) {
            return "FunnelModel";
        }
        if (normalized.contains("fnet")) {
            return "FNetModel";
        }
        if (normalized.contains("flaubert")) {
            return "FlaubertModel";
        }
        if (normalized.contains("mpnet")) {
            return "MPNetModel";
        }
        if (normalized.contains("xlnet")) {
            return "XLNetModel";
        }
        if (normalized.equals("xmod") || normalized.contains("xmod_") || normalized.contains("xmod-")) {
            return "XmodModel";
        }
        if (normalized.equals("xlm") || normalized.contains("xlm_") || normalized.contains("xlm-")) {
            return "XLMModel";
        }
        if (normalized.contains("electra")) {
            return "ElectraModel";
        }
        if (normalized.contains("albert")) {
            return "AlbertModel";
        }
        if (normalized.contains("roformer")) {
            return "RoFormerModel";
        }
        if (normalized.contains("deberta_v2") || normalized.contains("deberta-v2")) {
            return "DebertaV2Model";
        }
        if (normalized.contains("deberta")) {
            return "DebertaModel";
        }
        if (normalized.contains("esmfold")) {
            return "EsmForProteinFolding";
        }
        if (normalized.equals("esm") || normalized.contains("esm_") || normalized.contains("esm-")) {
            return "EsmModel";
        }
        if (normalized.contains("biogpt") || normalized.contains("bio_gpt")) {
            return "BioGptForCausalLM";
        }
        if (normalized.equals("opt") || normalized.contains("facebook_opt")) {
            return "OPTForCausalLM";
        }
        if (normalized.contains("ctrl")) {
            return "CTRLLMHeadModel";
        }
        if (normalized.contains("cpmant") || normalized.contains("cpm-ant")) {
            return "CpmAntForCausalLM";
        }
        if (normalized.equals("cpm") || normalized.contains("cpm_") || normalized.contains("cpm-")) {
            return "CpmForCausalLM";
        }
        if (normalized.contains("ernie")) {
            return "ErnieModel";
        }
        if (normalized.contains("canine")) {
            return "CanineModel";
        }
        if (normalized.contains("flava")) {
            return "FlavaModel";
        }
        
        // Default fallback
        return "LlamaForCausalLM";
    }
    
    public Optional<String> detectModelType(Map<String, Object> metadata) {
        Object value = metadata.get("general.architecture");
        if (value == null) {
            value = metadata.get("model_type");
        }
        if (value == null) {
            return Optional.empty();
        }
        String text = value.toString().trim();
        return text.isEmpty() ? Optional.empty() : Optional.of(text);
    }
}
