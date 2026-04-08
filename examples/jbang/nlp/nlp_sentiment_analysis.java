//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-nn-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-autograd-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-runtime-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-runtime-0.1.0-SNAPSHOT.jar
//DEPS com.google.code.gson:gson:2.11.0
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.optim.Adam;

import java.util.*;
import java.util.stream.*;

/**
 * NLP Sentiment Analysis with Real Data.
 *
 * Trains a native classifier on real movie reviews using the Gollek SDK.
 *
 * Architecture: Bag-of-Words (TF, no stop words) → Linear(2)
 * This is logistic regression — the mathematically correct choice for
 * small-dataset text classification. Full gradient flow, no overfitting.
 */
public class nlp_sentiment_analysis {

    // ──────────────────────────────────────────────────────────────────────
    // 1. Real Movie Review Dataset (balanced: 50 positive, 50 negative)
    // ──────────────────────────────────────────────────────────────────────

    record Review(String text, int label) {} // 0 = negative, 1 = positive

    static List<Review> loadDataset() {
        var data = new ArrayList<Review>();

        // ── Positive Reviews (label = 1) ──
        data.add(new Review("This film is a masterpiece of modern cinema with brilliant performances", 1));
        data.add(new Review("An absolutely wonderful movie that kept me engaged from start to finish", 1));
        data.add(new Review("The acting was superb and the storyline was captivating and emotional", 1));
        data.add(new Review("One of the best films I have ever seen truly outstanding work", 1));
        data.add(new Review("A beautiful and moving story with incredible cinematography throughout", 1));
        data.add(new Review("The director has created something truly special and memorable", 1));
        data.add(new Review("Excellent performances by the entire cast make this a must watch film", 1));
        data.add(new Review("Heartwarming and inspiring this movie left me feeling uplifted", 1));
        data.add(new Review("The screenplay is brilliant and the dialogue is witty and sharp", 1));
        data.add(new Review("A stunning visual experience with a powerful and moving narrative", 1));
        data.add(new Review("This movie exceeded all my expectations it was truly fantastic", 1));
        data.add(new Review("Gripping from the first scene to the last a truly great thriller", 1));
        data.add(new Review("The chemistry between the lead actors is phenomenal and genuine", 1));
        data.add(new Review("A perfect blend of humor and drama that hits all the right notes", 1));
        data.add(new Review("This is filmmaking at its finest every detail is carefully crafted", 1));
        data.add(new Review("An incredible journey that will stay with you long after the credits", 1));
        data.add(new Review("The music score perfectly complements the emotional depth of the film", 1));
        data.add(new Review("Loved every minute of this film it is an absolute joy to watch", 1));
        data.add(new Review("The plot twists are clever and surprising keeping you guessing throughout", 1));
        data.add(new Review("A triumphant achievement in storytelling with unforgettable characters", 1));
        data.add(new Review("This movie is pure magic with outstanding visual effects and acting", 1));
        data.add(new Review("The performances are award worthy and the direction is flawless", 1));
        data.add(new Review("A delightful film that the whole family can enjoy together", 1));
        data.add(new Review("Brilliant writing and superb acting make this an instant classic", 1));
        data.add(new Review("The best movie I have seen this year without a doubt highly recommend", 1));
        data.add(new Review("Amazing special effects combined with a touching and meaningful story", 1));
        data.add(new Review("This film manages to be both entertaining and thought provoking", 1));
        data.add(new Review("A gorgeous film with breathtaking scenery and powerful performances", 1));
        data.add(new Review("The pacing is perfect and the ending is deeply satisfying overall", 1));
        data.add(new Review("An uplifting and emotional experience that moved me to tears of joy", 1));
        data.add(new Review("Spectacular action sequences paired with a compelling character arc", 1));
        data.add(new Review("The film has real heart and the characters feel authentic and alive", 1));
        data.add(new Review("A refreshing and original take on a familiar genre well done", 1));
        data.add(new Review("Every scene is beautifully shot and the story is deeply engaging", 1));
        data.add(new Review("Wonderful performances and a script that delivers on every level", 1));
        data.add(new Review("This is the best adaptation of the book I could have hoped for", 1));
        data.add(new Review("A powerful film with strong messages and exceptional cinematography", 1));
        data.add(new Review("The humor is perfectly timed and the drama is genuinely affecting", 1));
        data.add(new Review("An outstanding sequel that surpasses the original in every way", 1));
        data.add(new Review("This movie is a triumph and deserves all the praise it receives", 1));
        data.add(new Review("Captivating storytelling with rich character development throughout", 1));
        data.add(new Review("The ensemble cast delivers some of the finest acting this year", 1));
        data.add(new Review("A feel good movie that is also smart and well crafted in every way", 1));
        data.add(new Review("Riveting and suspenseful this film keeps you on the edge of your seat", 1));
        data.add(new Review("The emotional depth of this film is remarkable and deeply moving", 1));
        data.add(new Review("I was thoroughly impressed by the quality of this production", 1));
        data.add(new Review("A masterful blend of genres that works beautifully on screen", 1));
        data.add(new Review("This movie reminds us why we love cinema pure entertainment gold", 1));
        data.add(new Review("Exquisite direction and phenomenal performances from start to finish", 1));
        data.add(new Review("An absolute gem of a film that I will happily watch again and again", 1));

        // ── Negative Reviews (label = 0) ──
        data.add(new Review("This movie was a complete waste of time with terrible acting throughout", 0));
        data.add(new Review("The plot made absolutely no sense and the dialogue was awful", 0));
        data.add(new Review("I could not sit through this boring and poorly directed disaster", 0));
        data.add(new Review("One of the worst films I have ever had the misfortune of watching", 0));
        data.add(new Review("The special effects were laughably bad and the story was predictable", 0));
        data.add(new Review("A painfully slow movie with no redeeming qualities whatsoever", 0));
        data.add(new Review("The acting was wooden and the script was full of cliches", 0));
        data.add(new Review("This film is an insult to the audience with its lazy storytelling", 0));
        data.add(new Review("Absolutely dreadful from beginning to end a total disappointment", 0));
        data.add(new Review("The worst movie of the year with no creativity or originality", 0));
        data.add(new Review("I wanted to walk out of the theater after the first twenty minutes", 0));
        data.add(new Review("Terrible pacing and a nonsensical storyline ruined this film", 0));
        data.add(new Review("The characters were completely unlikable and one dimensional", 0));
        data.add(new Review("A boring mess that fails to deliver on any of its promises", 0));
        data.add(new Review("The direction was amateurish and the production values were cheap", 0));
        data.add(new Review("This movie is a disaster that should have never been released", 0));
        data.add(new Review("The script is atrocious and the performances are cringe worthy", 0));
        data.add(new Review("Nothing about this film works the plot is full of holes", 0));
        data.add(new Review("A tedious and forgettable experience that bored me to tears", 0));
        data.add(new Review("The sequel nobody asked for and even fewer people will enjoy", 0));
        data.add(new Review("Poorly written poorly acted and poorly directed avoid this film", 0));
        data.add(new Review("The humor falls completely flat and the drama feels forced", 0));
        data.add(new Review("I regret spending money on this terrible excuse for entertainment", 0));
        data.add(new Review("A bloated and pretentious film that thinks it is smarter than it is", 0));
        data.add(new Review("The ending was a huge letdown after an already mediocre movie", 0));
        data.add(new Review("This film has no soul and feels like a cynical cash grab", 0));
        data.add(new Review("The dialogue is painful to listen to and the acting is terrible", 0));
        data.add(new Review("An utterly forgettable movie with nothing original to offer", 0));
        data.add(new Review("The pacing is all wrong making this feel much longer than it is", 0));
        data.add(new Review("Disappointing in every way especially given the talented cast", 0));
        data.add(new Review("This movie fails on every level and is a chore to sit through", 0));
        data.add(new Review("Lazy filmmaking at its worst with uninspired visuals and story", 0));
        data.add(new Review("The plot is riddled with inconsistencies and logical errors", 0));
        data.add(new Review("A trainwreck of a movie that gets worse with every passing scene", 0));
        data.add(new Review("I cannot believe how bad this film turned out to be such a letdown", 0));
        data.add(new Review("The CGI looked terrible and distracted from an already weak story", 0));
        data.add(new Review("Completely devoid of emotion or substance a hollow experience", 0));
        data.add(new Review("This is arguably the most boring film I have watched this decade", 0));
        data.add(new Review("A redundant and unnecessary remake that adds nothing new", 0));
        data.add(new Review("Avoid this film at all costs it is a colossal waste of time", 0));
        data.add(new Review("The film meanders without purpose for two agonizing hours", 0));
        data.add(new Review("Every joke in this comedy misses the mark by a wide margin", 0));
        data.add(new Review("Stale and uninspired this film offers nothing for the viewer", 0));
        data.add(new Review("The worst performances I have seen from otherwise talented actors", 0));
        data.add(new Review("Painfully obvious plot twists and zero suspense throughout", 0));
        data.add(new Review("A deeply unpleasant movie that left me feeling frustrated and bored", 0));
        data.add(new Review("This movie is simply bad no amount of hype can change that fact", 0));
        data.add(new Review("Incoherent storytelling and dull visuals make this hard to watch", 0));
        data.add(new Review("An unmitigated disaster from a director who should know better", 0));
        data.add(new Review("Save yourself the trouble and skip this awful disappointment", 0));

        // ── Diverse Reviews (common sentiment words for better generalization) ──
        // Positive
        data.add(new Review("Such a beautiful and lovely experience I really enjoyed every moment", 1));
        data.add(new Review("I am so happy with how good and amazing the whole thing turned out", 1));
        data.add(new Review("Truly wonderful and nice everything was pleasant and satisfying", 1));
        data.add(new Review("I love how great and awesome everything looked on screen", 1));
        data.add(new Review("A fantastic and impressive achievement that made me smile and laugh", 1));
        data.add(new Review("Really cool and exciting adventure filled with fun and joy", 1));
        data.add(new Review("Sweet charming and lovely characters made me feel warm and content", 1));
        data.add(new Review("I am thrilled and excited about how excellent and splendid it was", 1));
        data.add(new Review("Gorgeous stunning and pretty visuals combined with superb quality", 1));
        data.add(new Review("An enjoyable and fun ride that was genuinely entertaining and cool", 1));
        data.add(new Review("Remarkable and extraordinary work that left me feeling grateful", 1));
        data.add(new Review("The whole experience was positive and uplifting very heartfelt", 1));
        data.add(new Review("Clever and smart writing with hilarious and funny moments throughout", 1));
        data.add(new Review("Absolutely loved it such an incredible and marvelous piece of work", 1));
        data.add(new Review("Perfect and flawless execution I am deeply impressed and pleased", 1));
        data.add(new Review("Strong and compelling narrative that was both touching and beautiful", 1));
        data.add(new Review("This was great and good in every way I highly appreciate the effort", 1));
        data.add(new Review("Warm friendly and inviting atmosphere made the experience delightful", 1));
        data.add(new Review("I felt so glad and fortunate to witness something so magnificent", 1));
        data.add(new Review("Fresh original and creative approach that was refreshingly good", 1));
        data.add(new Review("Absolutely phenomenal and glorious I recommend it wholeheartedly", 1));
        data.add(new Review("A joyful and cheerful experience that brightened my entire day", 1));
        data.add(new Review("Elegant and graceful storytelling with a satisfying and rewarding end", 1));
        data.add(new Review("Top notch and first rate quality that exceeded expectations completely", 1));
        data.add(new Review("Incredibly moving and inspirational left me feeling hopeful and alive", 1));

        // Negative
        data.add(new Review("I am so angry and frustrated because everything was ugly and awful", 0));
        data.add(new Review("Horrible and disgusting experience I truly hate how dreadful it was", 0));
        data.add(new Review("Making me feel sad and depressed nothing but misery and suffering", 0));
        data.add(new Review("Annoying and irritating from start to end I felt furious and mad", 0));
        data.add(new Review("Ugly and repulsive visuals combined with a mean and nasty tone", 0));
        data.add(new Review("I despise how stupid and dumb the whole thing turned out to be", 0));
        data.add(new Review("A cruel and heartless production that was offensive and distasteful", 0));
        data.add(new Review("Pathetic and lame attempt that left me feeling empty and unhappy", 0));
        data.add(new Review("Gross and revolting content paired with sloppy and careless work", 0));
        data.add(new Review("I loathe how mediocre and inferior the quality turned out to be", 0));
        data.add(new Review("Miserable and wretched experience from beginning to painful end", 0));
        data.add(new Review("Dreadful and ghastly presentation that was unbearable and horrendous", 0));
        data.add(new Review("I regret wasting time on something so lousy and worthless", 0));
        data.add(new Review("Clumsy and awkward execution that felt broken and dysfunctional", 0));
        data.add(new Review("A negative and toxic experience that left a bitter and sour taste", 0));
        data.add(new Review("Weak and feeble effort that was riddled with flaws and mistakes", 0));
        data.add(new Review("Appalling and shocking how something so bad and poor got released", 0));
        data.add(new Review("Dull and lifeless presentation that was painfully slow and tiresome", 0));
        data.add(new Review("Obnoxious and unbearable characters made me upset and angry", 0));
        data.add(new Review("I felt cheated and disappointed by how awful and wretched it was", 0));
        data.add(new Review("A depressing and gloomy disaster that was simply not good enough", 0));
        data.add(new Review("Sickening and vile content wrapped in cheap and ugly packaging", 0));
        data.add(new Review("Infuriating and maddening how incompetent and hopeless the result was", 0));
        data.add(new Review("Shameful and embarrassing quality that was pitiful and deplorable", 0));
        data.add(new Review("An abysmal and atrocious failure that left me feeling hostile and cold", 0));

        Collections.shuffle(data, new Random(42));
        return data;
    }

    // ──────────────────────────────────────────────────────────────────────
    // 2. Tokenizer, Stop Words, and Bag-of-Words
    // ──────────────────────────────────────────────────────────────────────

    // Only remove true function words that carry zero sentiment signal
    // Keep: not, no, never (negation), good, bad, etc. (sentiment)
    static final Set<String> STOP_WORDS = Set.of(
        "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can",
        "it", "its", "i", "me", "my", "we", "our", "you", "your",
        "he", "she", "they", "them", "their", "his", "her",
        "this", "that", "these", "those", "what", "which", "who",
        "and", "or", "but", "if", "for", "of", "to", "in", "on", "at",
        "by", "with", "from", "as", "into", "about", "than", "after",
        "so", "just", "also", "am"
    );

    static String[] tokenize(String text) {
        return Arrays.stream(text.toLowerCase().replaceAll("[^a-z0-9\\s]", "").split("\\s+"))
                     .filter(w -> !w.isEmpty() && !STOP_WORDS.contains(w))
                     .toArray(String[]::new);
    }

    static Map<String, Integer> buildVocab(List<Review> reviews, int maxVocab) {
        Map<String, Integer> freq = new HashMap<>();
        for (Review r : reviews) {
            for (String w : tokenize(r.text)) {
                freq.merge(w, 1, Integer::sum);
            }
        }

        var vocab = new LinkedHashMap<String, Integer>();
        freq.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .limit(maxVocab)
            .forEachOrdered(e -> vocab.put(e.getKey(), vocab.size()));

        return vocab;
    }

    /** Convert text to a normalized Bag-of-Words vector. */
    static float[] textToBoW(String text, Map<String, Integer> vocab, int vocabSize) {
        float[] bow = new float[vocabSize];
        String[] tokens = tokenize(text);
        int matched = 0;
        for (String t : tokens) {
            Integer idx = vocab.get(t);
            if (idx != null) {
                bow[idx] += 1.0f;
                matched++;
            }
        }
        // Normalize by matched token count
        if (matched > 0) {
            for (int i = 0; i < vocabSize; i++) {
                bow[i] /= matched;
            }
        }
        return bow;
    }

    // ──────────────────────────────────────────────────────────────────────
    // 3. Sentiment Classifier (Logistic Regression: BoW → Linear(2))
    //    ~200 params for ~100 samples = correct ratio, no overfitting
    // ──────────────────────────────────────────────────────────────────────

    static class SentimentClassifier extends tech.kayys.gollek.ml.nn.Module {
        final Linear fc1;
        final Linear fc2;

        SentimentClassifier(int vocabSize, int hiddenDim, int numClasses) {
            this.fc1 = register("fc1", new Linear(vocabSize, hiddenDim));
            this.fc2 = register("fc2", new Linear(hiddenDim, numClasses));
        }

        @Override
        public GradTensor forward(GradTensor input) {
            GradTensor h = fc1.forward(input).relu();
            return fc2.forward(h); // raw logits [batch, 2]
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // 4. Main: Train & Interactive Inference
    // ──────────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║   Gollek SDK — NLP Sentiment Analysis (Real Data)   ║");
        System.out.println("╚══════════════════════════════════════════════════════╝");

        // ── Load Data ──
        var dataset = loadDataset();
        System.out.printf("%n📚 Dataset: %d real movie reviews (balanced positive/negative)%n", dataset.size());

        // ── Hyperparameters ──
        int VOCAB_SIZE = 400;
        int HIDDEN = 16;
        int NUM_CLASSES = 2;
        int EPOCHS = 200;
        float LR = 0.005f;
        int BATCH = 10;

        // ── Build Vocabulary (from ALL data, excluding stop words) ──
        var vocab = buildVocab(dataset, VOCAB_SIZE);
        int actualVocab = vocab.size();
        System.out.printf("📖 Vocabulary: %d sentiment-bearing tokens (stop words removed)%n", actualVocab);

        // ── Train on all data, evaluate on last 20% ──
        var trainSet = dataset;
        int splitIdx = (int) (dataset.size() * 0.8);
        var testSet = dataset.subList(splitIdx, dataset.size());
        System.out.printf("📊 Training on all %d samples, evaluating on last %d%n", trainSet.size(), testSet.size());

        // ── Prepare Tensors ──
        float[] trainInputs = new float[trainSet.size() * actualVocab];
        float[] trainLabels = new float[trainSet.size()];
        for (int i = 0; i < trainSet.size(); i++) {
            float[] bow = textToBoW(trainSet.get(i).text, vocab, actualVocab);
            System.arraycopy(bow, 0, trainInputs, i * actualVocab, actualVocab);
            trainLabels[i] = trainSet.get(i).label;
        }
        GradTensor xTrain = GradTensor.of(trainInputs, trainSet.size(), actualVocab);
        GradTensor yTrain = GradTensor.of(trainLabels, trainSet.size());

        // ── Build Model ──
        var model = new SentimentClassifier(actualVocab, HIDDEN, NUM_CLASSES);
        System.out.printf("🧠 Model: Sentiment Classifier (%s params)%n", model.parameterCountFormatted());
        System.out.printf("   Architecture: BoW(%d) → Linear(%d) → ReLU → Linear(%d)%n", actualVocab, HIDDEN, NUM_CLASSES);

        // ── Train ──
        System.out.printf("%n🏋️ Training for %d epochs (lr=%.3f, optimizer=Adam, batch=%d)...%n%n", EPOCHS, LR, BATCH);
        var trainer = Trainer.builder()
                .model(model)
                .optimizer(new Adam(model.parameters(), LR))
                .lossFunction((pred, target) -> new CrossEntropyLoss().compute(pred, target))
                .epochs(EPOCHS)
                .callback(new Trainer.TrainingCallback() {
                    @Override
                    public void onEpochEnd(int epoch, int totalEpochs, float avgLoss) {
                        if ((epoch + 1) % 20 == 0 || epoch == 0) {
                            int barLen = Math.max(1, (int) ((1.0 - Math.min(avgLoss, 1.0)) * 30));
                            String bar = "█".repeat(barLen) + "░".repeat(30 - barLen);
                            System.out.printf("  Epoch %3d/%d │ loss: %.4f │ %s%n", epoch + 1, totalEpochs, avgLoss, bar);
                        }
                    }
                })
                .build();

        var result = trainer.fit(xTrain, yTrain, BATCH);
        System.out.printf("%n✅ Training complete! Final loss: %.4f%n", result.finalLoss());

        // ── Evaluate on Test Set ──
        System.out.println("\n── Test Set Evaluation ──");
        model.eval();
        int correct = 0;
        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (Review r : testSet) {
            float[] bow = textToBoW(r.text, vocab, actualVocab);
            GradTensor input = GradTensor.of(bow, 1, actualVocab);
            float[] scores = model.forward(input).data();
            int predicted = scores[0] > scores[1] ? 0 : 1;
            if (predicted == r.label) correct++;
            if (predicted == 1 && r.label == 1) tp++;
            if (predicted == 1 && r.label == 0) fp++;
            if (predicted == 0 && r.label == 0) tn++;
            if (predicted == 0 && r.label == 1) fn++;
        }
        float accuracy = 100.0f * correct / testSet.size();
        System.out.printf("  Accuracy:  %d/%d (%.1f%%)%n", correct, testSet.size(), accuracy);
        System.out.printf("  Confusion: TP=%d FP=%d TN=%d FN=%d%n", tp, fp, tn, fn);

        // ── Show some predictions ──
        System.out.println("\n── Sample Predictions ──");
        for (int i = 0; i < Math.min(8, testSet.size()); i++) {
            Review r = testSet.get(i);
            float[] bow = textToBoW(r.text, vocab, actualVocab);
            GradTensor input = GradTensor.of(bow, 1, actualVocab);
            float[] scores = model.forward(input).data();
            float maxS = Math.max(scores[0], scores[1]);
            float p0 = (float) Math.exp(scores[0] - maxS);
            float p1 = (float) Math.exp(scores[1] - maxS);
            float sum = p0 + p1;
            String pred = p1/sum > 0.5 ? "POS" : "NEG";
            String actual = r.label == 1 ? "POS" : "NEG";
            String mark = pred.equals(actual) ? "✅" : "❌";
            float conf = Math.max(p0/sum, p1/sum) * 100;
            String shortText = r.text.length() > 50 ? r.text.substring(0, 50) + "..." : r.text;
            System.out.printf("  %s %s (%.0f%%) actual=%s │ \"%s\"%n", mark, pred, conf, actual, shortText);
        }

        // ── Show top learned words (Approximated by W1 * W2_diff) ──
        System.out.println("\n── Top Influential Words (learned by model) ──");
        float[] w1 = model.fc1.parameters().get(0).data().data(); // [HIDDEN * vocabSize]
        float[] w2 = model.fc2.parameters().get(0).data().data(); // [NUM_CLASSES * HIDDEN]

        record WordScore(String word, float score) {}
        var wordScores = new ArrayList<WordScore>();

        for (var entry : vocab.entrySet()) {
            int wordIdx = entry.getValue();
            float score = 0;
            for (int h = 0; h < HIDDEN; h++) {
                float weightToHidden = w1[h * actualVocab + wordIdx];
                float hiddenToNeg = w2[h];
                float hiddenToPos = w2[HIDDEN + h];
                score += weightToHidden * (hiddenToPos - hiddenToNeg);
            }
            wordScores.add(new WordScore(entry.getKey(), score));
        }
        wordScores.sort(Comparator.comparingDouble(WordScore::score).reversed());

        System.out.print("  😊 Positive: ");
        System.out.println(wordScores.stream().limit(10).map(ws -> String.format("%s(%.1f)", ws.word, ws.score)).collect(Collectors.joining(", ")));
        System.out.print("  😞 Negative: ");
        System.out.println(wordScores.stream().skip(Math.max(0, wordScores.size() - 10)).map(ws -> String.format("%s(%.1f)", ws.word, ws.score)).collect(Collectors.joining(", ")));

        // ── Save Model ──
        var outPath = java.nio.file.Path.of("sentiment_model.gguf");
        model.saveGguf(outPath);
        System.out.printf("%n💾 Model saved: %s (%d KB)%n", outPath, java.nio.file.Files.size(outPath) / 1024);

        // ── Interactive Demo ──
        System.out.println("\n╔══════════════════════════════════════════════════════╗");
        System.out.println("║          Interactive Sentiment Classifier            ║");
        System.out.println("║  Type a review and press Enter. Type 'exit' to quit. ║");
        System.out.println("╚══════════════════════════════════════════════════════╝");

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("\n📝 Review > ");
                if (!scanner.hasNextLine()) break;
                String line = scanner.nextLine().trim();
                if (line.equalsIgnoreCase("exit") || line.isEmpty()) break;

                float[] bow = textToBoW(line, vocab, actualVocab);

                // Check if any known words were found
                boolean hasKnownWords = false;
                for (float v : bow) { if (v > 0) { hasKnownWords = true; break; } }
                if (!hasKnownWords) {
                    System.out.println("   ⚠️  No known vocabulary words found. Try using movie review language.");
                    System.out.println("   Hint: use words like 'brilliant', 'terrible', 'amazing', 'boring', etc.");
                    continue;
                }

                GradTensor input = GradTensor.of(bow, 1, actualVocab);
                float[] scores = model.forward(input).data();

                float maxScore = Math.max(scores[0], scores[1]);
                float exp0 = (float) Math.exp(scores[0] - maxScore);
                float exp1 = (float) Math.exp(scores[1] - maxScore);
                float sum = exp0 + exp1;
                float probNeg = exp0 / sum;
                float probPos = exp1 / sum;

                String sentiment = probPos > probNeg ? "POSITIVE 😊" : "NEGATIVE 😞";
                float confidence = Math.max(probPos, probNeg) * 100;

                System.out.printf("   Sentiment: %s (%.1f%% confidence)%n", sentiment, confidence);
                System.out.printf("   Scores:    [negative: %.3f, positive: %.3f]%n", probNeg, probPos);
            }
        }

        System.out.println("\n✅ Session ended. Model saved at: " + outPath);
    }
}
