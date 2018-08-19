package models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.Map.Entry;

import utility.FuncUtils;
import utility.LBFGS;
import utility.Parallel;
import cc.mallet.optimize.InvalidOptimizableException;
import cc.mallet.optimize.Optimizer;
import cc.mallet.types.MatrixOps;

public class ASTM {
	public double alpha; // Hyper-parameter alpha
	public double beta; // Hyper-parameter alpha
	public double alphaSum; // alpha * numTopics
	public double betaSum; // beta * vocabularySize

	public int numTopics; // Number of topics
	public int topWords; // Number of most probable words for each topic

	public int numInitIterations;
	public int numIterations; // Number of EM-style sampling iterations

	public List<List<Integer>> corpus; // Word ID-based corpus
	public List<List<Integer>> topicAssignments; // Topics assignments for words
													// in the corpus
	public int numDocuments; // Number of documents in the corpus
	public int numWordsInCorpus; // Number of words in the corpus

	public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
														// given a word
	public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
														// given an ID
	public int vocabularySize; // The number of word types in the corpus

	// numDocuments * numTopics matrix
	// Given a document: number of its words assigned to each topic
	public int[][] docTopicCount;
	// Number of words in every document
	public int[] sumDocTopicCount;
	// numTopics * vocabularySize matrix
	// Given a topic: number of times a word type generated from the topic by
	// the Dirichlet multinomial component
	public int[][] topicWordCountLDA;
	// Total number of words generated from each topic by the Dirichlet
	// multinomial component
	public int[] sumTopicWordCountLDA;

	// Double array used to sample a topic
	/**
	 * 前LF 后LDA
	 */
	public double[] multiPros;
	// Path to the directory containing the corpus
	public String folderPath;
	// Path to the topic modeling corpus
	public String corpusPath;
	public String vectorFilePath;

	/**
	 * wordVectors[wordID][词向量的维]
	 */
	public double[][] wordVectors; // Vector representations for words
	public double[][] topicVectors;// Vector representations for topics
	public int vectorSize; // Number of vector dimensions
	public double[][] dotProductValues;
	public double[][] expDotProductValues;// [t][w]is a vector of “scores”
											// indexed by words?
	public double[] sumExpValues; // Partition function values

	public final double l2Regularizer = 0.01; // L2 regularizer value for
												// learning topic vectors
	public final double tolerance = 0.05; // Tolerance value for LBFGS//
											// convergence
	/**
	 * Ss: 哈哈哈哈 出了6个回响
	 */
	public int[] outWords;
	double[][] reverPhi;
	public int sumWords = 0;
	double[][] pzw;
	Random rg = new Random();
	double segThreshold = 0.25;

	public SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

	// 需要点什么呢
	public int[] segTopicCount;
	// Number of words in every segmentation
	public int[] sumSegTopicCount;
	// numTopics * vocabularySize matrix
	// Given a topic: number of times a word type generated from the topic by
	// the Dirichlet multinomial component
	public int[][] topicWordCountDMM;
	// Total number of words generated from each topic
	public int[] sumTopicWordCountDMM;
	public List<List<Integer>> segmentations = new ArrayList<>();
	public List<List<Integer>> segAssignments = new ArrayList<>();

	// corpusSeg.get(d) == segID in doc
	public List<List<Integer>> corpusSeg = new ArrayList<>();

	public String expName = "AS";
	public String orgExpName = "AS";
	public String tAssignsFilePath = "";
	public int savestep = 0;

	public ASTM(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics, double inAlpha, double inBeta,
			double inLambda, int inNumInitIterations, int inNumIterations, int inTopWords, String inExpName)
			throws Exception {
		this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda, inNumInitIterations,
				inNumIterations, inTopWords, inExpName, "", 0);
	}

	/***
	 * 读入dataset 在无pathToTAfile的情况下 随机初始化 调用 initialize
	 */
	public ASTM(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics, double inAlpha, double inBeta,
			double inLambda, int inNumInitIterations, int inNumIterations, int inTopWords, String inExpName,
			String pathToTAfile, int inSaveStep) throws Exception {
		alpha = inAlpha;
		beta = inBeta;
		numTopics = inNumTopics;
		numIterations = inNumIterations;
		numInitIterations = inNumInitIterations;
		topWords = inTopWords;
		savestep = inSaveStep;
		expName = inExpName;
		orgExpName = expName;
		vectorFilePath = pathToWordVectorsFile;
		corpusPath = pathToCorpus;
		folderPath = pathToCorpus.substring(0,
				Math.max(pathToCorpus.lastIndexOf("/"), pathToCorpus.lastIndexOf("\\")) + 1);

		System.out.println("Reading topic modeling corpus: " + pathToCorpus);

		word2IdVocabulary = new HashMap<String, Integer>();
		id2WordVocabulary = new HashMap<Integer, String>();
		corpus = new ArrayList<List<Integer>>();
		numDocuments = 0;
		numWordsInCorpus = 0;

		HashSet<String> embeddingSet = getWordswithEmbedding(pathToWordVectorsFile);

		// load data set
		BufferedReader br = null;
		try {
			int indexWord = -1;
			br = new BufferedReader(new FileReader(pathToCorpus));
			HashSet<String> outList = new HashSet<>();
			for (String doc; (doc = br.readLine()) != null;) {

				if (doc.trim().length() == 0)
					continue;

				doc = doc.replaceAll("\\d+ ", "");
				String[] words = doc.trim().split("\\s+");
				List<Integer> document = new ArrayList<Integer>();

				for (String word : words) {
					if (!embeddingSet.contains(word)) {
						outList.add(word);
					}
					if (word2IdVocabulary.containsKey(word)) {
						document.add(word2IdVocabulary.get(word));
					} else {
						indexWord += 1;
						word2IdVocabulary.put(word, indexWord);
						id2WordVocabulary.put(indexWord, word);
						document.add(indexWord);
					}
				}

				numDocuments++;
				numWordsInCorpus += document.size();
				corpus.add(document);
			}
			outWords = new int[outList.size()];
			System.out.println("out of vocabulary: " + outList.size());
			int i = 0;
			for (String w : outList) {
				outWords[i++] = word2IdVocabulary.get(w);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		// init
		vocabularySize = word2IdVocabulary.size();
		System.out.println(vocabularySize);
		docTopicCount = new int[numDocuments][numTopics];
		sumDocTopicCount = new int[numDocuments];
		topicWordCountLDA = new int[numTopics][vocabularySize];
		sumTopicWordCountLDA = new int[numTopics];

		multiPros = new double[numTopics];
		for (int i = 0; i < numTopics; i++) {
			multiPros[i] = 1.0 / numTopics;
		}

		alphaSum = numTopics * alpha;
		betaSum = vocabularySize * beta;

		// 词向量的读入
		readWordVectorsFile(vectorFilePath);
		topicVectors = new double[numTopics][vectorSize];
		dotProductValues = new double[numTopics][vocabularySize];
		expDotProductValues = new double[numTopics][vocabularySize];
		sumExpValues = new double[numTopics];

		System.out.println("Corpus size: " + numDocuments + " docs, " + numWordsInCorpus + " words");
		System.out.println("Vocabuary size: " + vocabularySize);
		System.out.println("Number of topics: " + numTopics);
		System.out.println("alpha: " + alpha);
		System.out.println("beta: " + beta);
		System.out.println("Number of initial sampling iterations: " + numInitIterations);
		System.out.println("Number of EM-style sampling iterations for the LF-LDA model: " + numIterations);
		System.out.println("Number of top topical words: " + topWords);
		initialize();
	}

	//
	public HashSet<String> getWordswithEmbedding(String pathToWordVectorsFile) {
		HashSet<String> wordswithEmbedding = new HashSet<>();

		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(pathToWordVectorsFile));

			for (String line; (line = br.readLine()) != null;) {
				String[] elements = line.trim().split("\\s+");
				String word = elements[0];
				wordswithEmbedding.add(word);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return wordswithEmbedding;
	}

	public void readWordVectorsFile(String pathToWordVectorsFile) throws Exception {
		System.out.println("Reading word vectors from word-vectors file " + pathToWordVectorsFile + "...");

		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(pathToWordVectorsFile));
			String[] elements = br.readLine().trim().split("\\s+");
			vectorSize = elements.length - 1;
			wordVectors = new double[vocabularySize][vectorSize];
			String word = elements[0];
			if (word2IdVocabulary.containsKey(word)) {
				for (int j = 0; j < vectorSize; j++) {
					wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
				}
			}
			for (String line; (line = br.readLine()) != null;) {
				elements = line.trim().split("\\s+");
				word = elements[0];
				if (word2IdVocabulary.containsKey(word)) {
					for (int j = 0; j < vectorSize; j++) {
						wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		for (int i = 0; i < vocabularySize; i++) {
			if (MatrixOps.absNorm(wordVectors[i]) == 0.0) {
			}
		}
	}

	public void initialize() throws IOException {
		System.out.println("Randomly initialzing topic assignments ...");
		topicAssignments = new ArrayList<List<Integer>>();

		for (int docId = 0; docId < numDocuments; docId++) {
			List<Integer> topics = new ArrayList<Integer>();
			int docSize = corpus.get(docId).size();
			for (int j = 0; j < docSize; j++) {
				int wordId = corpus.get(docId).get(j);

				int topic = FuncUtils.nextDiscrete(multiPros);
				topicWordCountLDA[topic][wordId] += 1;
				sumTopicWordCountLDA[topic] += 1;
				docTopicCount[docId][topic] += 1;
				sumDocTopicCount[docId] += 1;

				topics.add(topic);
			}
			topicAssignments.add(topics);
		}
	}

	public void initalInference() throws IOException {
		System.out.println("Running Gibbs sampling inference: ");
		System.out.println("segT: " + segThreshold);

		// numInitIterations == 1000 avoid cold start
		for (int iter = 1; iter <= numInitIterations; iter++) {

			if (iter % 100 == 0) {
				System.out.println(df.format(System.currentTimeMillis()));
				System.out.println("\tInitial sampling iteration: " + (iter));
			}

			sampleLDAInitialIteration();
		}

		double[] countwords = new double[vocabularySize];
		for (int w = 0; w < vocabularySize; w++) {
			for (int i = 0; i < numTopics; i++) {
				countwords[w] += topicWordCountLDA[i][w];
			}
		}
		// ③ p(z|w)
		pzw = new double[vocabularySize][numTopics];
		for (int w = 0; w < vocabularySize; w++) {
			double rowsum = 0.0;
			for (int t = 0; t < numTopics; t++) {
				pzw[w][t] = 1.0 * (topicWordCountLDA[t][w] + beta) / (countwords[w] + beta * numTopics);
			}
		}
		for (int outid : outWords) {
			wordVectors[outid] = sampleOneWordEmbedding(outid);
		}


		for (int iter = 1; iter <= numIterations; iter++) {

			if (iter % 100 == 0) {
				System.out.println(df.format(System.currentTimeMillis()));
				System.out.println("\tJointly sampling iteration: " + (iter));
			}

			optimizeTopicVectors(true);
			sampleAttentionalIteration(); 
		}
		expName = orgExpName;

	}

	public void initalSegmentation() {
		try {
			for (int dIndex = 0; dIndex < numDocuments; dIndex++) { // doc
				List<Integer> cs = new ArrayList<>();// this corpus contains
														// those segments
				int docSize = corpus.get(dIndex).size();
				double[] scores = new double[docSize - 1];
				String tesentence = "";
				for (int wIndex = 0; wIndex < docSize - 1; wIndex++) {
					int word = corpus.get(dIndex).get(wIndex);// wordID
					int nextword = corpus.get(dIndex).get(wIndex + 1);

					double[] ctemp = new double[numTopics];
					double[] ntemp = new double[numTopics];
					for (int tIndex = 0; tIndex < numTopics; tIndex++) {
						ctemp[tIndex] = expDotProductValues[tIndex][word] / sumExpValues[tIndex];
						ntemp[tIndex] = expDotProductValues[tIndex][nextword] / sumExpValues[tIndex];
					}
					tesentence += id2WordVocabulary.get(word) + " ";
					double s = FuncUtils.calCosSimilarity(ctemp, ntemp);
					scores[wIndex] = s;

				}
				
				ArrayList<Integer> aSegment = new ArrayList<>();
				ArrayList<Integer> aSegTopic = new ArrayList<>();
				int ftopic = topicAssignments.get(dIndex).get(0);
				aSegment.add(corpus.get(dIndex).get(0));
				for (int i = 0; i < scores.length; i++) {
					if (scores[i] > segThreshold) {
						aSegment.add(corpus.get(dIndex).get(i + 1));
						continue;
					} else {
						segmentations.add(aSegment);
						cs.add(segmentations.size() - 1);
						for (int tt : aSegment)
							aSegTopic.add(ftopic);
						segAssignments.add(aSegTopic);

						aSegment = new ArrayList<>();
						aSegTopic = new ArrayList<>();
						aSegment.add(corpus.get(dIndex).get(i + 1));
						ftopic = topicAssignments.get(dIndex).get(i + 1);
						continue;
					}
				}
				segmentations.add(aSegment);
				cs.add(segmentations.size() - 1);
				for (int tt : aSegment)
					aSegTopic.add(ftopic);
				segAssignments.add(aSegTopic);


				corpusSeg.add(cs);
				
			}
			initDMM();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void initDMM() throws IOException {
		System.out.println("initialzing count matrixs ...");
		topicWordCountDMM = new int[numTopics][vocabularySize];
		sumTopicWordCountDMM = new int[numTopics];
		segTopicCount = new int[numTopics];

		// for two matrices
		for (int segId = 0; segId < segmentations.size(); segId++) {
			int topic = segAssignments.get(segId).get(0);
			segTopicCount[topic] += 1;
			for (int j = 0; j < segmentations.get(segId).size(); j++) {
				int wordId = segmentations.get(segId).get(j);
				topicWordCountDMM[topic][wordId] += 1;
				sumTopicWordCountDMM[topic] += 1;
			}
		}
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) { // doc
			ArrayList<Integer> temp = new ArrayList<>();
			for (int sid : corpusSeg.get(dIndex)) {
				for (int j = 0; j < segAssignments.get(sid).size(); j++) {
					temp.add(segAssignments.get(sid).get(j));
				}
			}
			topicAssignments.set(dIndex, temp);
		}


	}

	public void sampleDMMIteration() {
		for (int sIndex = 0; sIndex < segmentations.size(); sIndex++) {
			List<Integer> segment = segmentations.get(sIndex);
			int topic = segAssignments.get(sIndex).get(0);

			segTopicCount[topic] = segTopicCount[topic] - 1;
			for (int wIndex = 0; wIndex < segment.size(); wIndex++) {
				int word = segment.get(wIndex);
				topicWordCountDMM[topic][word] -= 1;
				sumTopicWordCountDMM[topic] -= 1;
			}

			// Sample a topic
			for (int tIndex = 0; tIndex < numTopics; tIndex++) {
				multiPros[tIndex] = (segTopicCount[tIndex] + alpha);
				for (int wIndex = 0; wIndex < segment.size(); wIndex++) {
					int word = segment.get(wIndex);
					double signal = expDotProductValues[tIndex][word] / sumExpValues[tIndex];
					multiPros[tIndex] *= signal * (topicWordCountDMM[tIndex][word] + beta)
							/ (sumTopicWordCountDMM[tIndex] + betaSum);
				}
			}
			topic = FuncUtils.nextDiscrete(multiPros);

			segTopicCount[topic] += 1;
			for (int wIndex = 0; wIndex < segment.size(); wIndex++) {
				int word = segment.get(wIndex);// wordID
				topicWordCountDMM[topic][word] += 1;
				sumTopicWordCountDMM[topic] += 1;
				// Update topic assignments
				segAssignments.get(sIndex).set(wIndex, topic);
			}
		}
	}

	public void inference() throws IOException {
		for (int iter = 1; iter <= 500; iter++) {

			if (iter % 500 == 0)
				System.out.println("\tFinal sampling iteration: " + (iter));

			sampleDMMIteration();

			if (iter % 100 == 0) {

				optimizeTopicVectors(false);
				updateSegmentaion();
			}
		}
		expName = orgExpName + "-" + segThreshold;
		finalWrite();
	}

	public void updateSegmentaion() {

		for (int dIndex = 0; dIndex < numDocuments; dIndex++) { // doc
			ArrayList<Integer> temp = new ArrayList<>();
			for (int sid : corpusSeg.get(dIndex)) {
				for (int j = 0; j < segAssignments.get(sid).size(); j++) {
					temp.add(segAssignments.get(sid).get(j));
				}
			}
			topicAssignments.set(dIndex, temp);
		}
		segmentations = new ArrayList<>();
		segAssignments = new ArrayList<>();
		corpusSeg = new ArrayList<>();

		for (int dIndex = 0; dIndex < numDocuments; dIndex++) { // doc
			List<Integer> cs = new ArrayList<>();// this corpus contains
													// those segments
			int docSize = corpus.get(dIndex).size();
			double[] scores = new double[docSize - 1];
			String tesentence = "";
			for (int wIndex = 0; wIndex < docSize - 1; wIndex++) {
				int word = corpus.get(dIndex).get(wIndex);// wordID
				int nextword = corpus.get(dIndex).get(wIndex + 1);

				double[] ctemp = new double[numTopics];
				double[] ntemp = new double[numTopics];
				for (int tIndex = 0; tIndex < numTopics; tIndex++) {
					ctemp[tIndex] = expDotProductValues[tIndex][word] / sumExpValues[tIndex];
					ntemp[tIndex] = expDotProductValues[tIndex][nextword] / sumExpValues[tIndex];
				}
				tesentence += id2WordVocabulary.get(word) + " ";
				double s = FuncUtils.calCosSimilarity(ctemp, ntemp);
				scores[wIndex] = s;

			}

			ArrayList<Integer> aSegment = new ArrayList<>();
			ArrayList<Integer> aSegTopic = new ArrayList<>();
			aSegment.add(corpus.get(dIndex).get(0));
			int ftopic = 0;
			for (int i = 0; i < scores.length; i++) {
				ftopic = topicAssignments.get(dIndex).get(i);
				if (scores[i] > segThreshold) {
					aSegment.add(corpus.get(dIndex).get(i + 1));
					continue;
				} else {
					segmentations.add(aSegment);
					cs.add(segmentations.size() - 1);
					for (int tt : aSegment)
						aSegTopic.add(ftopic);
					segAssignments.add(aSegTopic);

					aSegment = new ArrayList<>();
					aSegTopic = new ArrayList<>();
					aSegment.add(corpus.get(dIndex).get(i + 1));
					ftopic = topicAssignments.get(dIndex).get(i + 1);
					continue;
				}
			}
			segmentations.add(aSegment);
			cs.add(segmentations.size() - 1);
			for (int tt : aSegment)
				aSegTopic.add(ftopic);
			segAssignments.add(aSegTopic);

			corpusSeg.add(cs);
		}

	}

	public double[][] calPhibyLDA(boolean flag) {
		double[][] phi = new double[numTopics][vocabularySize];
		for (int w = 0; w < vocabularySize; w++) {
			for (int t = 0; t < numTopics; t++) {
				if (flag)
					phi[t][w] = (topicWordCountLDA[t][w] + beta) / (sumTopicWordCountLDA[t] + betaSum);
				else
					phi[t][w] = (topicWordCountDMM[t][w] + beta) / (sumTopicWordCountDMM[t] + betaSum);
			}
		}
		return phi;
	}
	public void optimizeTopicVectors(boolean flag) {

		sumExpValues = new double[numTopics];
		dotProductValues = new double[numTopics][vocabularySize];
		expDotProductValues = new double[numTopics][vocabularySize];

		double[][] phi = calPhibyLDA(flag);

		Parallel.loop(numTopics, new Parallel.LoopInt()// end = numTopics
		{
			@Override
			public void compute(int topic)// The int topic is from 0 to
											// numTopics
			{
				int rate = 1;
				boolean check = true;
				while (check) {
					double l2Value = l2Regularizer * rate;
					try {
						TopicVectorOptimizer optimizer = new TopicVectorOptimizer(topicVectors[topic], phi[topic],
								wordVectors, l2Value);

						Optimizer gd = new LBFGS(optimizer, tolerance);
						gd.optimize(600);
						optimizer.getParameters(topicVectors[topic]);
						sumExpValues[topic] = optimizer.computePartitionFunction(dotProductValues[topic],
								expDotProductValues[topic]);
						check = false;
						if (sumExpValues[topic] == 0 || Double.isInfinite(sumExpValues[topic])) {
							double max = -1000000000.0;
							for (int index = 0; index < vocabularySize; index++) {
								if (dotProductValues[topic][index] > max)
									max = dotProductValues[topic][index];
							}
							for (int index = 0; index < vocabularySize; index++) {
								expDotProductValues[topic][index] = Math.exp(dotProductValues[topic][index] - max);
								sumExpValues[topic] += expDotProductValues[topic][index];
							}
						}
					} catch (InvalidOptimizableException e) {
						e.printStackTrace();
						check = true;
					}
					rate = rate * 10;
				}
			}
		});
	}


	public void sampleLDAInitialIteration() {
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				int word = corpus.get(dIndex).get(wIndex);// wordID
				int topic = topicAssignments.get(dIndex).get(wIndex);

				docTopicCount[dIndex][topic] -= 1;
				topicWordCountLDA[topic][word] -= 1;
				sumTopicWordCountLDA[topic] -= 1;

				// Sample a pair of topic z and binary indicator variable s
				for (int tIndex = 0; tIndex < numTopics; tIndex++) {

					multiPros[tIndex] = (docTopicCount[dIndex][tIndex] + alpha)
							* (topicWordCountLDA[tIndex][word] + beta) / (sumTopicWordCountLDA[tIndex] + betaSum);

				}
				topic = FuncUtils.nextDiscrete(multiPros);

				docTopicCount[dIndex][topic] += 1;
				topicWordCountLDA[topic][word] += 1;
				sumTopicWordCountLDA[topic] += 1;
				// Update topic assignments
				topicAssignments.get(dIndex).set(wIndex, topic);
			}

		}
	}

	public void sampleAttentionalIteration() {
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				int word = corpus.get(dIndex).get(wIndex);// wordID
				int topic = topicAssignments.get(dIndex).get(wIndex);

				docTopicCount[dIndex][topic] -= 1;
				topicWordCountLDA[topic][word] -= 1;
				sumTopicWordCountLDA[topic] -= 1;

				// Sample a pair of topic z and binary indicator variable s
				for (int tIndex = 0; tIndex < numTopics; tIndex++) {
					double signal = expDotProductValues[tIndex][word] / sumExpValues[tIndex];
					// signal = 1.0;
					multiPros[tIndex] = signal * (docTopicCount[dIndex][tIndex] + alpha)
							* (topicWordCountLDA[tIndex][word] + beta) / (sumTopicWordCountLDA[tIndex] + betaSum);

				}
				topic = FuncUtils.nextDiscrete(multiPros);

				docTopicCount[dIndex][topic] += 1;
				topicWordCountLDA[topic][word] += 1;
				sumTopicWordCountLDA[topic] += 1;
				// Update topic assignments
				topicAssignments.get(dIndex).set(wIndex, topic);
			}

		}
	}

	public double[] calOneWordEmbedding(int outid) {
		double[] embedding = new double[vectorSize];

		double[] features = pzw[outid];
		HashMap<String, Double> simiMap = new HashMap<>();
		for (int i = 0; i < vocabularySize; i++) {
			simiMap.put(id2WordVocabulary.get(i), getSimilarity(features, pzw[i]));
		}

		
		List<Entry<String, Double>> list = new ArrayList<Entry<String, Double>>(simiMap.entrySet());

		Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
			public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
				double p = o1.getValue() - o2.getValue();
				if (p > 0) {
					return 1;
				} else if (p == 0) {
					return 0;
				} else
					return -1;
			}
		});

		for (int i = 0; i < 10; i++) {
			int wid = word2IdVocabulary.get(list.get(i).getKey());
			for (int j = 0; j < vectorSize; j++) {
				embedding[j] += wordVectors[wid][j];
			}
		}

		// avg
		for (int j = 0; j < vectorSize; j++) {
			embedding[j] = embedding[j] / 10;
		}
		return embedding;
	}

	public double[] sampleOneWordEmbedding(int outid) {
		double[] embedding = new double[vectorSize];
		double[] features = pzw[outid];
		int count = 0;
		for (int i = 0; i < vocabularySize; i++) {
			double distance = getSimilarity(features, pzw[i]);
			double a = rg.nextDouble();
			if (Double.compare(distance, a) < 0) { //
				count++;
				for (int j = 0; j < vectorSize; j++) {
					embedding[j] += wordVectors[i][j];
				}
			}
		}
		// avg
		for (int j = 0; j < vectorSize; j++) {
			embedding[j] = embedding[j] / count;
		}
		return embedding;
	}

	public double getSimilarity(double[] a, double[] b) {
		double simi = 0;
		double sum = 0;
		double avg = 0, temp = 0;
		for (int i = 0; i < a.length; i++) {
			avg = (a[i] + b[i]) / 2;
			temp = Math.log(a[i] / avg) / Math.log(2);
			// temp = Math.log(a[i] /avg);
			simi += a[i] * temp;
			temp = Math.log(b[i] / avg) / Math.log(2);
			// temp = Math.log(b[i] /avg);
			simi += b[i] * temp;

		}
		simi = simi * 0.5;
		// System.out.println(simi);
		return simi;
	}

	public void writeParameters() throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".paras"));
		writer.write("-model" + "\t" + "LFLDA");
		writer.write("\n-corpus" + "\t" + corpusPath);
		writer.write("\n-vectors" + "\t" + vectorFilePath);
		writer.write("\n-ntopics" + "\t" + numTopics);
		writer.write("\n-alpha" + "\t" + alpha);
		writer.write("\n-beta" + "\t" + beta);
		writer.write("\n-initers" + "\t" + numInitIterations);
		writer.write("\n-niters" + "\t" + numIterations);
		writer.write("\n-twords" + "\t" + topWords);
		writer.write("\n-name" + "\t" + expName);
		if (tAssignsFilePath.length() > 0)// 这个没有执行啊
			writer.write("\n-initFile" + "\t" + tAssignsFilePath);
		if (savestep > 0)
			writer.write("\n-sstep" + "\t" + savestep);

		writer.close();
	}

	public void writeDictionary() throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".vocabulary"));
		for (String word : word2IdVocabulary.keySet()) {
			writer.write(word + " " + word2IdVocabulary.get(word) + "\n");
		}
		writer.close();
	}

	public void writeIDbasedCorpus() throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".IDcorpus"));
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				writer.write(corpus.get(dIndex).get(wIndex) + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	/**
	 * file : ".topicAssignments"
	 * 
	 * @throws IOException
	 */
	public void writeTopicAssignments() throws IOException {

		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".topicAssignments"));
		for (int d = 0; d < corpusSeg.size(); d++) {
			for (int s = 0; s < corpusSeg.get(d).size(); s++) {
				int segID = corpusSeg.get(d).get(s); // segments in Doc d
				int topic = segAssignments.get(segID).get(0);
				for (int tt : segmentations.get(segID)) {// numWord in this
															// segment
					writer.write(topic + " ");
				}
				//
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopicVectors() throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".topicEmbedding"));

		Map<Integer, Double> topicWordProbs = new TreeMap<Integer, Double>();
		for (int i = 0; i < numTopics; i++) {
			for (int wIndex = 0; wIndex < vocabularySize; wIndex++) {

				double pro = (topicWordCountLDA[i][wIndex] + beta) / (sumTopicWordCountLDA[i] + betaSum);

				topicWordProbs.put(wIndex, pro);
			}
			topicWordProbs = FuncUtils.sortByValueDescending(topicWordProbs);

			Set<Integer> mostLikelyWords = topicWordProbs.keySet();
			int count = 0;
			for (Integer index : mostLikelyWords) {
				if (count < topWords) {
					writer.write(" " + id2WordVocabulary.get(index));
					count += 1;
				} else {
					writer.write("\n\n");
					break;
				}
			}

			for (int j = 0; j < vectorSize; j++)
				writer.write(topicVectors[i][j] + " ");
			writer.write("\n");
		}
		writer.close();
	}


	public void writeFinalTopTopicalWords() throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".finalTopWords"));

		for (int tIndex = 0; tIndex < numTopics; tIndex++) {
			writer.write("Topic" + new Integer(tIndex) + ":");

			Map<Integer, Double> topicWordProbs = new TreeMap<Integer, Double>();
			for (int wIndex = 0; wIndex < vocabularySize; wIndex++) {

				double pro = (topicWordCountDMM[tIndex][wIndex] + beta) / (sumTopicWordCountDMM[tIndex] + betaSum);

				topicWordProbs.put(wIndex, pro);
			}
			topicWordProbs = FuncUtils.sortByValueDescending(topicWordProbs);

			Set<Integer> mostLikelyWords = topicWordProbs.keySet();
			int count = 0;
			for (Integer index : mostLikelyWords) {
				if (count < topWords) {
					writer.write(" " + id2WordVocabulary.get(index));
					count += 1;
				} else {
					writer.write("\n\n");
					break;
				}
			}
		}
		writer.close();
	}

	/**
	 * file: ".phi"
	 * 
	 * @throws IOException
	 */
	public void writeTopicWordPros() throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".LDAphi"));
		for (int t = 0; t < numTopics; t++) {
			for (int w = 0; w < vocabularySize; w++) {
				double pro = (topicWordCountLDA[t][w] + beta) / (sumTopicWordCountLDA[t] + betaSum);
				writer.write(pro + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	/**
	 * file: ".theta"
	 * 
	 * @throws IOException
	 */
	public void writeDocTopicPros() throws IOException {

		// sampling changed.
		// calculate doc-t docTopicCount[d][t]
		docTopicCount = new int[numDocuments][numTopics];
		sumDocTopicCount = new int[numDocuments];
		System.out.println(corpusSeg.size());
		for (int d = 0; d < corpusSeg.size(); d++) {
			for (int s = 0; s < corpusSeg.get(d).size(); s++) {
				int segID = corpusSeg.get(d).get(s); // segments in Doc d
				int topic = segAssignments.get(segID).get(0);
				for (int tt : segmentations.get(segID)) {// numWord in this
															// segment
					docTopicCount[d][topic]++;
					sumDocTopicCount[d]++;
				}
				//
			}
		}

		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".theta"));
		for (int i = 0; i < numDocuments; i++) {
			for (int j = 0; j < numTopics; j++) {
				double pro = (docTopicCount[i][j] + alpha) / (sumDocTopicCount[i] + alphaSum);
				writer.write(pro + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}


	public void writeWordEmbedding(int iter) throws IOException {
		FileWriter fileWriter1 = new FileWriter(folderPath + expName + "-" + iter + ".tsv");
		FileWriter fileWriter2 = new FileWriter(folderPath + expName + "-" + iter + ".labeltsv");
		for (int i = 0; i < vocabularySize; i++) {
			for (double d : wordVectors[i]) {
				fileWriter1.write(d + "\t");
			}
			fileWriter1.write("\r\n");
			fileWriter2.write(id2WordVocabulary.get(i) + "\r\n");
		}
		fileWriter1.close();
		fileWriter2.close();
	}



	public void finalWrite() throws IOException {
		writeFinalTopTopicalWords();
		writeDocTopicPros();
		writeTopicAssignments();
		writeTopicVectors();
	}

	public static void main(String args[]) throws Exception {
		String embeddingPath = "";
		String textPath = "";
		String outname = "";
		String fname = "";
		int numtopic = 5;
		double lamda = 0;
		for (File file : new File("data/baby").listFiles()) {
			if (file.getName().endsWith(".embedding")) {
				embeddingPath = file.getPath();
			}
			if (file.getName().endsWith(".input")) {
				textPath = file.getPath();
				fname = file.getName();
			}
		}

		int[] as = { 5, 10, 20, 30, 50, 70, 90, 100 };// the number of topics
		double[] bs = { 0.5};//threshold 
		for (int i : as) {
			numtopic = i;
			for (double j : bs) {
				lamda = 1;
				for (int ii = 0; ii < 10; ii++) {
					outname = fname + "-ASTM-" + numtopic + "-tt-" + j + "-ii-" + ii;
					ASTM astm = new ASTM(textPath, embeddingPath, numtopic, 0.1, 0.01, lamda, 1000, 500, 20, outname);
					astm.segThreshold = j;
					astm.initalInference();
					astm.initalSegmentation();
					astm.inference();
				}
			}
		}

	}
}
