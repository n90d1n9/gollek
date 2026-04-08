//REPOS mavenLocal
//DEPS tech.kayys.gollek:gollek-sdk-litert:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-runner-litert:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-spi-inference:0.1.0-SNAPSHOT
//JAVA 21+

import tech.kayys.gollek.sdk.litert.LiteRTSdk;
import tech.kayys.gollek.sdk.litert.config.LiteRTConfig;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.Attachment;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.util.Base64;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import javax.imageio.ImageIO;

/**
 * LiteRT Image Classification Example
 * 
 * Demonstrates image classification using LiteRT models like MobileNet,
 * EfficientNet, etc.
 * 
 * Usage:
 * jbang ImageClassificationExample.java --model mobilenet.litertlm --image
 * cat.jpg
 * jbang ImageClassificationExample.java --model mobilenet.litertlm --image
 * cat.jpg --topk 5
 * jbang ImageClassificationExample.java --model mobilenet.litertlm --image
 * cat.jpg --delegate GPU
 * 
 * @author Gollek Team
 * @version 0.1.0
 */
public class ImageClassificationExample {

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║        LiteRT Image Classification Example               ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        // Parse arguments
        String modelPath = System.getProperty("model", "mobilenet_v2.litertlm");
        String imagePath = System.getProperty("image", null);
        int topK = Integer.parseInt(System.getProperty("topk", "5"));
        String delegateStr = System.getProperty("delegate", "AUTO");

        if (imagePath == null) {
            System.err.println("Error: --image parameter is required");
            System.err.println(
                    "Usage: jbang ImageClassificationExample.java --model <model.litertlm> --image <image.jpg>");
            System.exit(1);
        }

        LiteRTConfig.Delegate delegate = LiteRTConfig.Delegate.valueOf(delegateStr.toUpperCase());

        // Load image
        System.out.println("Loading image: " + imagePath);
        BufferedImage image = ImageIO.read(new File(imagePath));
        if (image == null) {
            System.err.println("Error: Could not load image: " + imagePath);
            System.exit(1);
        }
        System.out.println("  Size: " + image.getWidth() + "x" + image.getHeight());
        System.out.println();

        // Create SDK
        LiteRTConfig config = LiteRTConfig.builder()
                .numThreads(4)
                .delegate(delegate)
                .enableXnnpack(true)
                .useMemoryPool(true)
                .build();

        try (LiteRTSdk sdk = new LiteRTSdk(config)) {
            // Load model
            System.out.println("Loading model: " + modelPath);
            sdk.loadModel("classifier", Path.of(modelPath));
            System.out.println("✓ Model loaded");
            System.out.println();

            // Preprocess image
            System.out.println("Preprocessing image...");
            byte[] inputData = preprocessImage(image, 224, 224);
            System.out.println("  Input size: " + inputData.length + " bytes");
            System.out.println();

            // Run inference
            System.out.println("Running inference...");
            long startTime = System.currentTimeMillis();

            InferenceRequest request = InferenceRequest.builder()
                    .model("classifier")
                    .message(Message.user("classify"))
                    .attachment(Attachment.fromBase64(Base64.getEncoder().encodeToString(inputData), "application/octet-stream"))
                    .build();

            InferenceResponse response = sdk.infer(request);
            long elapsed = System.currentTimeMillis() - startTime;

            System.out.println("✓ Inference completed in " + elapsed + "ms");
            System.out.println();

            // Post-process results
            System.out.println("Top " + topK + " Predictions:");
            byte[] outputData = (byte[]) response.getMetadata().get("output_tensor");
            if (outputData != null) {
                printTopK(outputData, topK);
            } else {
                System.out.println("  Warning: No output tensor found in response metadata.");
                System.out.println("  Content: " + response.getContent());
            }
            System.out.println();

            // Run batch classification with multiple crops
            System.out.println("Running batch classification (5 crops)...");
            batchClassification(sdk, image);
        }
    }

    /**
     * Preprocess image for model input.
     * Resizes to target size and converts to float32 tensor.
     */
    private static byte[] preprocessImage(BufferedImage image, int targetWidth, int targetHeight) {
        // Resize image
        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        resized.getGraphics().drawImage(image, 0, 0, targetWidth, targetHeight, null);

        // Convert to float32 tensor [1, height, width, 3]
        ByteBuffer buffer = ByteBuffer.allocate(1 * targetHeight * targetWidth * 3 * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                int rgb = resized.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                // Normalize to [-1, 1] (common for MobileNet)
                buffer.putFloat((r / 127.5f) - 1.0f);
                buffer.putFloat((g / 127.5f) - 1.0f);
                buffer.putFloat((b / 127.5f) - 1.0f);
            }
        }

        return buffer.array();
    }

    /**
     * Print top K predictions.
     */
    private static void printTopK(byte[] outputData, int topK) {
        // Convert output to float array
        ByteBuffer buffer = ByteBuffer.wrap(outputData);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        int numClasses = buffer.capacity() / 4;
        float[] probabilities = new float[numClasses];

        for (int i = 0; i < numClasses; i++) {
            probabilities[i] = buffer.getFloat();
        }

        // Find top K indices
        int[] topIndices = new int[topK];
        float[] topValues = new float[topK];

        for (int i = 0; i < numClasses; i++) {
            float prob = probabilities[i];
            int minIdx = 0;
            for (int j = 1; j < topK; j++) {
                if (topValues[j] < topValues[minIdx]) {
                    minIdx = j;
                }
            }

            if (prob > topValues[minIdx]) {
                topIndices[minIdx] = i;
                topValues[minIdx] = prob;
            }
        }

        // Sort by probability
        Integer[] sortedIndices = new Integer[topK];
        for (int i = 0; i < topK; i++)
            sortedIndices[i] = i;
        java.util.Arrays.sort(sortedIndices, (a, b) -> Float.compare(topValues[b], topValues[a]));

        // Print results
        for (int i = 0; i < topK; i++) {
            int idx = sortedIndices[i];
            int classIdx = topIndices[idx];
            float prob = topValues[idx];

            String label = getLabel(classIdx);
            System.out.printf("  %d. %-30s %.2f%%%n", i + 1, label, prob * 100);
        }
    }

    /**
     * Run batch classification with multiple image crops.
     */
    private static void batchClassification(LiteRTSdk sdk, BufferedImage image) throws Exception {
        int numCrops = 5;
        int cropSize = 224;

        java.util.List<InferenceRequest> requests = new java.util.ArrayList<>();

        // Generate multiple crops
        for (int i = 0; i < numCrops; i++) {
            // Extract crop (simplified - just use full image for demo)
            byte[] cropData = preprocessImage(image, cropSize, cropSize);

            InferenceRequest request = InferenceRequest.builder()
                    .model("classifier")
                    .message(Message.user("classify"))
                    .attachment(Attachment.fromBase64(Base64.getEncoder().encodeToString(cropData), "application/octet-stream"))
                    .build();

            requests.add(request);
        }

        // Run batch inference
        long startTime = System.currentTimeMillis();
        java.util.List<InferenceResponse> responses = sdk.inferBatch(requests);
        long elapsed = System.currentTimeMillis() - startTime;

        System.out.println("  Processed " + responses.size() + " crops in " + elapsed + "ms");
        System.out.println("  Average per crop: " + (elapsed / numCrops) + "ms");
        System.out.println();
    }

    /**
     * Get label for class index.
     * ImageNet 1000 class labels for models like MobileNet, EfficientNet, etc.
     */
    private static String getLabel(int classIndex) {
        // ImageNet ILSVRC 2012 1000-class labels
        String[] labels = {
            "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
            "electric ray", "stingray", "cock", "hen", "ostrich",
            "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
            "robin", "bulbul", "jay", "magpie", "chickadee",
            "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
            "European fire salamander", "common newt", "eft", "spotted salamander", "axolotl",
            "bullfrog", "tree frog", "tailed frog", "loggerhead", "leatherback turtle",
            "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana",
            "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard",
            "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile",
            "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake",
            "green snake", "king snake", "garter snake", "water snake", "vine snake",
            "night snake", "boa constrictor", "rock python", "Indian cobra", "green mamba",
            "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite",
            "harvestman", "scorpion", "black and gold garden spider", "barn spider", "garden spider",
            "black widow", "tarantula", "wolf spider", "tick", "centipede",
            "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock",
            "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo",
            "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
            "jacamar", "toucan", "drake", "red-breasted merganser", "goose",
            "black swan", "tusker", "echidna", "platypus", "wallaby",
            "koala", "wombat", "jellyfish", "sea anemone", "brain coral",
            "flatworm", "nematode", "conch", "snail", "slug",
            "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
            "fiddler crab", "king crab", "American lobster", "spiny lobster", "crayfish",
            "hermit crab", "isopod", "white stork", "black stork", "spoonbill",
            "flamingo", "little blue heron", "American egret", "bittern", "crane (bird)",
            "limpkin", "European gallinule", "American coot", "bustard", "ruddy turnstone",
            "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican",
            "king penguin", "albatross", "grey whale", "killer whale", "dugong",
            "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese",
            "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier", "Rhodesian ridgeback",
            "Afghan hound", "basset", "beagle", "bloodhound", "bluetick",
            "black-and-tan coonhound", "Walker hound", "English foxhound", "redbone", "borzoi",
            "Irish wolfhound", "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound",
            "otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier",
            "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier",
            "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier", "Lakeland terrier",
            "Sealyham terrier", "Airedale", "cairn", "Australian terrier", "Dandie Dinmont",
            "Boston bull", "miniature schnauzer", "giant schnauzer", "standard schnauzer", "Scotch terrier",
            "Tibetan terrier", "silky terrier", "soft-coated wheaten terrier", "West Highland white terrier", "Lhasa",
            "flat-coated retriever", "curly-coated retriever", "golden retriever", "Labrador retriever", "Chesapeake Bay retriever",
            "German short-haired pointer", "vizsla", "English setter", "Irish setter", "Gordon setter",
            "Brittany spaniel", "clumber", "English springer", "Welsh springer spaniel", "cocker spaniel",
            "Sussex spaniel", "Irish water spaniel", "kuvasz", "schipperke", "groenendael",
            "malinois", "briard", "kelpie", "komondor", "Old English sheepdog",
            "Shetland sheepdog", "collie", "Border collie", "Bouvier des Flandres", "Rottweiler",
            "German shepherd", "Doberman", "miniature pinscher", "Greater Swiss Mountain dog", "Bernese mountain dog",
            "Appenzeller", "EntleBucher", "boxer", "bull mastiff", "Tibetan mastiff",
            "French bulldog", "Great Dane", "Saint Bernard", "Eskimo dog", "malamute",
            "Siberian husky", "dalmatian", "affenpinscher", "basenji", "pug",
            "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian",
            "chow", "keeshond", "Brabancon griffon", "Pembroke", "Cardigan",
            "toy poodle", "miniature poodle", "standard poodle", "Mexican hairless", "timber wolf",
            "white wolf", "red wolf", "coyote", "dingo", "dhole",
            "African hunting dog", "hyena", "red fox", "kit fox", "Arctic fox",
            "grey fox", "tabby", "tiger cat", "Persian cat", "Siamese cat",
            "Egyptian cat", "cougar", "lynx", "leopard", "snow leopard",
            "jaguar", "lion", "tiger", "cheetah", "brown bear",
            "American black bear", "ice bear", "sloth bear", "mongoose", "meerkat",
            "tiger beetle", "ladybug", "ground beetle", "long-horned beetle", "leaf beetle",
            "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee",
            "ant", "grasshopper", "cricket", "walking stick", "cockroach",
            "mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
            "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly",
            "sulphur butterfly", "lycaenid", "starfish", "sea urchin", "sea cucumber",
            "wood rabbit", "hare", "Angora", "hamster", "porcupine",
            "fox squirrel", "marmot", "beaver", "guinea pig", "sorrel",
            "zebra", "hog", "wild boar", "warthog", "hippopotamus",
            "ox", "water buffalo", "bison", "ram", "bighorn",
            "ibex", "hartebeest", "impala", "gazelle", "Arabian camel",
            "llama", "weasel", "mink", "polecat", "black-footed ferret",
            "otter", "skunk", "badger", "armadillo", "three-toed sloth",
            "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang",
            "guenon", "patas", "baboon", "macaque", "langur",
            "colobus", "proboscis monkey", "marmoset", "capuchin", "howler monkey",
            "titi", "spider monkey", "squirrel monkey", "Madagascar cat", "indri",
            "Indian elephant", "African elephant", "lesser panda", "giant panda", "barracouta",
            "eel", "coho", "rock beauty", "anemone fish", "sturgeon",
            "gar", "lionfish", "puffer", "abacus", "abaya",
            "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner",
            "airship", "altar", "ambulance", "amphibian", "analog clock",
            "apiary", "apron", "ashcan", "assault rifle", "backpack",
            "bakery", "balance beam", "balloon", "ballpoint", "Band Aid",
            "banjo", "bannister", "barbell", "barber chair", "barbershop",
            "barn", "barometer", "barrel", "barrow", "baseball",
            "basketball", "bassinet", "bassoon", "bathing cap", "bath towel",
            "bathtub", "beach wagon", "beacon", "beaker", "bearskin",
            "beer bottle", "beer glass", "bell cote", "bib", "bicycle-built-for-two",
            "bikini", "binder", "binoculars", "birdhouse", "boathouse",
            "bobsled", "bolo tie", "bonnet", "bookcase", "bookshop",
            "bottlecap", "bow", "bow tie", "brass", "brassiere",
            "breakwater", "breastplate", "broom", "bucket", "buckle",
            "bulletproof vest", "bullet train", "butcher shop", "cab", "caldron",
            "candle", "cannon", "canoe", "can opener", "cardigan",
            "car mirror", "carousel", "carpenter's kit", "carton", "car wheel",
            "cash machine", "cassette", "cassette player", "castle", "catamaran",
            "CD player", "cello", "cellular telephone", "chain", "chainlink fence",
            "chain mail", "chain saw", "chest", "chiffonier", "chime",
            "china cabinet", "Christmas stocking", "church", "cinema", "cleaver",
            "cliff dwelling", "cloak", "clog", "cocktail shaker", "coffee mug",
            "coffeepot", "coil", "combination lock", "computer keyboard", "confectionery",
            "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
            "cowboy hat", "cradle", "crane (machine)", "crash helmet", "crate",
            "crib", "Crock Pot", "croquet ball", "crutch", "cuirass",
            "dam", "desk", "desktop computer", "dial telephone", "diaper",
            "digital clock", "digital watch", "dining table", "dishrag", "dishwasher",
            "disk brake", "dock", "dogsled", "dome", "doormat",
            "drilling platform", "drum", "drumstick", "dumbbell", "Dutch oven",
            "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope",
            "espresso maker", "face powder", "feather boa", "file", "fireboat",
            "fire engine", "fire screen", "flagpole", "flute", "folding chair",
            "football helmet", "forklift", "fountain", "fountain pen", "four-poster",
            "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
            "gasmask", "gas pump", "goblet", "go-kart", "golf ball",
            "golfcart", "gondola", "gong", "gown", "grand piano",
            "greenhouse", "grille", "grocery store", "guillotine", "hair slide",
            "hair spray", "half track", "hammer", "hamper", "hand blower",
            "hand-held computer", "handkerchief", "hard disc", "harmonica", "harp",
            "harvester", "hatchet", "holster", "home theater", "honeycomb",
            "hook", "hoopskirt", "horizontal bar", "horse cart", "hourglass",
            "iPod", "iron", "jack-o'-lantern", "jean", "jeep",
            "jersey", "jigsaw puzzle", "jinrikisha", "joystick", "kimono",
            "knee pad", "knot", "lab coat", "ladle", "lampshade",
            "laptop", "lawn mower", "lens cap", "letter opener", "library",
            "lifeboat", "lighter", "limousine", "liner", "lipstick",
            "Loafer", "lotion", "loudspeaker", "loupe", "lumbermill",
            "magnetic compass", "mailbag", "mailbox", "maillot", "maillot (tank suit)",
            "manhole cover", "maraca", "marimba", "mask", "matchstick",
            "maypole", "maze", "measuring cup", "medicine chest", "megalith",
            "microphone", "microwave", "military uniform", "milk can", "minibus",
            "miniskirt", "minivan", "missile", "mitten", "mixing bowl",
            "mobile home", "Model T", "modem", "monastery", "monitor",
            "moped", "mortar", "mortarboard", "mosque", "mosquito net",
            "motor scooter", "mountain bike", "mountain tent", "mouse", "mousetrap",
            "moving van", "muzzle", "nail", "neck brace", "necklace",
            "nipple", "notebook", "obelisk", "oboe", "ocarina",
            "odometer", "oil filter", "organ", "oscilloscope", "overskirt",
            "oxcart", "oxygen mask", "packet", "paddle", "paddlewheel",
            "padlock", "paintbrush", "pajama", "palace", "panpipe",
            "paper towel", "parachute", "parallel bars", "park bench", "parking meter",
            "passenger car", "patio", "pay-phone", "pedestal", "pencil box",
            "pencil sharpener", "perfume", "Petri dish", "photocopier", "pick",
            "pickelhaube", "picket fence", "pickup", "pier", "piggy bank",
            "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate",
            "pitcher", "plane", "planetarium", "plastic bag", "plate rack",
            "plow", "plunger", "Polaroid camera", "pole", "police van",
            "poncho", "pool table", "pop bottle", "pot", "potter's wheel",
            "power drill", "prayer rug", "printer", "prison", "projectile",
            "projector", "puck", "punching bag", "purse", "quill",
            "quilt", "racer", "racket", "radiator", "radio",
            "radio telescope", "rain barrel", "recreational vehicle", "reel", "reflex camera",
            "refrigerator", "remote control", "restaurant", "revolver", "rifle",
            "rocking chair", "rotisserie", "rubber eraser", "rugby ball", "rule",
            "running shoe", "safe", "safety pin", "saltshaker", "sandal",
            "sarong", "sax", "scabbard", "scale", "school bus",
            "schooner", "scoreboard", "screen", "screw", "screwdriver",
            "seat belt", "sewing machine", "shield", "shoe shop", "shoji",
            "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain",
            "ski", "ski mask", "sleeping bag", "slide rule", "sliding door",
            "slot", "snorkel", "snowmobile", "snowplow", "soap dispenser",
            "soccer ball", "sock", "solar dish", "sombrero", "soup bowl",
            "space bar", "space heater", "space shuttle", "spatula", "speedboat",
            "spider web", "spindle", "sports car", "spotlight", "stage",
            "steam locomotive", "steel arch bridge", "steel drum", "stethoscope", "stole",
            "stone wall", "stopwatch", "stove", "strainer", "streetcar",
            "stretcher", "studio couch", "stupa", "submarine", "suit",
            "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge",
            "swab", "sweatshirt", "swimming trunks", "swing", "switch",
            "syringe", "table lamp", "tank", "tape player", "teapot",
            "teddy", "television", "tennis ball", "thatch", "theater curtain",
            "thimble", "thresher", "throne", "tile roof", "toaster",
            "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck",
            "toyshop", "tractor", "trailer truck", "tray", "trench coat",
            "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus",
            "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella",
            "unicycle", "upright", "vacuum", "vase", "vault",
            "velvet", "vending machine", "vestment", "viaduct", "violin",
            "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe",
            "warplane", "washbasin", "washer", "water bottle", "water jug",
            "water tower", "whiskey jug", "whistle", "wig", "window screen",
            "window shade", "Windsor tie", "wine bottle", "wing", "wok",
            "wooden spoon", "wool", "worm fence", "wreck", "yawl",
            "yurt", "web site", "comic book", "crossword puzzle", "street sign",
            "traffic light", "book jacket", "menu", "plate", "guacamole",
            "consomme", "hot pot", "trifle", "ice cream", "ice lolly",
            "French loaf", "bagel", "pretzel", "cheeseburger", "hotdog",
            "mashed potato", "head cabbage", "broccoli", "cauliflower", "zucchini",
            "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke",
            "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry",
            "orange", "lemon", "fig", "pineapple", "banana",
            "jackfruit", "custard apple", "pomegranate", "hay", "carbonara",
            "chocolate sauce", "dough", "meat loaf", "pizza", "potpie",
            "burrito", "red wine", "espresso", "cup", "eggnog",
            "alp", "bubble", "cliff", "coral reef", "geyser",
            "lakeside", "promontory", "sandbar", "seashore", "valley",
            "volcano", "ballplayer", "groom", "scuba diver", "rapeseed",
            "daisy", "yellow lady's slipper", "corn", "acorn", "hip",
            "buckeye", "coral fungus", "agaric", "gyromitra", "stinkhorn",
            "earthstar", "hen-of-the-woods", "bolete", "ear", "toilet tissue"
        };

        if (classIndex >= 0 && classIndex < labels.length) {
            return labels[classIndex];
        }
        return "Class " + classIndex;
    }
}
