//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// general includes

#include <time.h>
#include <stdlib.h>

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// for RL toolbox
#include "ril_debug.h"
#include "ctdlearner.h"
#include "cpolicies.h"
#include "cagent.h"
#include "cagentlogger.h"
#include "crewardmodel.h"
#include "canalyzer.h"
#include "cgridworldmodel.h"
#include "cvetraces.h"
#include "cvfunctionlearner.h"

#include "cadaptivesoftmaxnetwork.h"
#include "crbftrees.h"
//#include "ctorchvfunction.h"
#include "ccontinuousactions.h"
#include "MLP.h"
#include "GradientMachine.h"
#include "LogRBF.h"
#include "Linear.h"
#include "Tanh.h"
#include "RBFBasedQFunction.h"
#include "HashTable.h"

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// for multiboost
#include "Defaults.h"
#include "Utils/Args.h"

#include "WeakLearners/BaseLearner.h" // To get the list of the registered weak learners
#include "IO/Serialization.h" // for unserialization
#include "Bandits/GenericBanditAlgorithm.h"
#include "AdaBoostMDPClassifierAdv.h"
#include "AdaBoostMDPClassifierContinous.h"

using namespace std;
using namespace MultiBoost;
using namespace Torch;

#define SEP setw(15)

//#define LOGQTABLE
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

static const char CURRENT_VERSION[] = "1.5.00";


//---------------------------------------------------------------------------

/**
 * Check if a given base learner has been registered. If not it will give an error
 * and exit.
 * \param baseLearnerName The name of the base learner to be checked.
 * \date 21/3/2006
 */
void checkBaseLearner(const string& baseLearnerName)
{
	if ( !BaseLearner::RegisteredLearners().hasLearner(baseLearnerName) )
	{
		// Not found in the registered!
		cerr << "ERROR: learner <" << baseLearnerName << "> not found in the registered learners!" << endl;
		exit(1);
	}
}

//---------------------------------------------------------------------------

/**
 * Show the basic output. Called when no argument is provided.
 * \date 11/11/2005
 */
void showBase()
{
	cout << "MDDAG (v" << CURRENT_VERSION << "). Markov Decision Directed Acyclic Graph." << endl;
	cout << "---------------------------------------------------------------------------" << endl;
	cout << "Build: " << __DATE__ << " (" << __TIME__ << ")" << endl << endl;
	cout << "===> Type --help for help or --static to show the static options" << endl;
	
	exit(0);
}

//---------------------------------------------------------------------------

/**
 * Show the help. Called when -h argument is provided.
 * \date 11/11/2005
 */
void showHelp(nor_utils::Args& args, const vector<string>& learnersList)
{
	cout << "MDDAG (v" << CURRENT_VERSION << "). Markov Decision Directed Acyclic Graph." << endl;
	cout << "------------------------ HELP SECTION --------------------------" << endl;
	
	args.printGroup("Parameters");
	
	cout << endl;
	cout << "For specific help options type:" << endl;
	cout << "   --h general: General options" << endl;
	cout << endl;
	exit(0);
}

//---------------------------------------------------------------------------

/**
 * Show the help for the options.
 * \param args The arguments structure.
 * \date 28/11/2005
 */
void showOptionalHelp(nor_utils::Args& args)
{
	string helpType = args.getValue<string>("h", 0);
	
	cout << "MDDAG (v" << CURRENT_VERSION << ")." << endl;
	cout << "---------------------------------------------------------------------------" << endl;
	
	if (helpType == "general")
		args.printGroup("General Options");
	else if (helpType == "io")
		args.printGroup("I/O Options");
	else if (helpType == "algo")
		args.printGroup("Basic Algorithm Options");
	else if ( BaseLearner::RegisteredLearners().hasLearner(helpType) )
		args.printGroup(helpType + " Options");
	else
		cerr << "ERROR: Unknown help section <" << helpType << ">" << endl;
}

//---------------------------------------------------------------------------

/**
 * Show the default values.
 * \date 11/11/2005
 */
void showStaticConfig()
{
	cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
	cout << "------------------------ STATIC CONFIG -------------------------" << endl;
	
	cout << "- Sort type = ";
#if CONSERVATIVE_SORT
	cout << "CONSERVATIVE (slow)" << endl;
#else
	cout << "NON CONSERVATIVE (fast)" << endl;
#endif
	
	cout << "Comment: " << COMMENT << endl;
#ifndef NDEBUG
	cout << "Important: NDEBUG not active!!" << endl;
#endif
	
#if MB_DEBUG
	cout << "MultiBoost debug active (MB_DEBUG=1)!!" << endl;
#endif
	
	exit(0);
}

//---------------------------------------------------------------------------

void setBasicOptions(nor_utils::Args& args)
{
	args.setArgumentDiscriminator("--");
	
	args.declareArgument("help");
	args.declareArgument("static");
	
	args.declareArgument("h", "Help", 1, "<optiongroup>");
	
	//////////////////////////////////////////////////////////////////////////
	// Basic Arguments
	
	args.setGroup("Parameters");
    args.declareArgument("configfile", "Read some or all the argument from a config file.", 1, "<config file>");
	args.declareArgument("traintestmdp", "Performs training and test at the same time.", 5, "<trainingDataFile> <testDataFile> <nInterations> <shypfile> <outfile>");
    args.declareArgument("traintestmdp", "Performs training and test at the same time.", 6, "<trainingDataFile> <validDataFile> <nInterations> <shypfile> <outfile> <testDataFile>");
    args.declareArgument("testmdp", "Performs test of a previously leant model.", 3, "<qtable> <train log file> <test log file>");
    args.declareArgument("deeparff", "Outputs an arff file where the attributes are the paths of MDDAG.", 4, "<qtable> <train arff file> <test arff file> <mode>");
    
    
	args.declareArgument( "fileformat", "Defines the type of intput file. Available types are:\n"
						 "* simple: each line has attributes separated by whitespace and class at the end (DEFAULT!)\n"
						 "* arff: arff filetype. The header file can be specified using --arffheader option\n"
						 "* arffbzip: bziped arff filetype. The header file can be specified using --arffheader option\n"
						 "* svmlight: \n"
						 "(Example: --fileformat simple)",
                         1, "<fileFormat>" );
	
	args.declareArgument("headerfile", "The filename of the header file (SVMLight).", 1, "header.txt");
	
	
	args.setGroup("General Options");
	
	args.declareArgument("verbose", "Set the verbose level 0, 1 or 2 (0=no messages, 1=default, 2=all messages).", 1, "<val>");
	args.declareArgument("seed", "Defines the seed for the random operations.", 1, "<seedval>");
	
	//////////////////////////////////////////////////////////////////////////
	// Options for TL tool
	args.setGroup("RL options");
	
	args.declareArgument("episodes", "The number of episodes", 2, "<episod> <testiter>");
	args.declareArgument("rewards", "success, class, skip", 3, "<succ> <class> <skip>");
	args.declareArgument("logdir", "Dir of log", 1, "<dir>");
    args.declareArgument("qdir", "Dir of Q information", 1, "<dir>");
	args.declareArgument("succrewartdtype", "The mode of the reward calculation", 1, "<mode>" );
	args.declareArgument("statespace", "The statespace representation", 1, "<mode>" );
	args.declareArgument("numoffeat", "The number of feature in statespace representation", 1, "<featnum>" );
    args.declareArgument("optimistic", "Set the initial values of the Q function", 3, "<real> <real> <real>" );
    args.declareArgument("etrace", "Lambda parameter", 1, "<real>" );
    args.declareArgument("rbfbias", "Set the bias of the RBF network in the QTable representation", 3, "<real> <real> <real>" );
    args.declareArgument("noaddcenter", "Disactivate adding centers on TD error", 0, "" );
    args.declareArgument("normrbf", "Normalized RBFs", 0, "" );
    args.declareArgument("rbfsigma", "Initialize RBF with a given sigma", 1, "<val>" );
    args.declareArgument("maxtderr", "Max error on the TD value for adding a center. It is given by the inverse of the ratio of the max Q", 1, "<val>" );
    args.declareArgument("minrbfact", "Min activation factor for adding a center", 1, "<val>" );
    args.declareArgument("positivelabel", "The name of positive label", 1, "<labelname>" );
    args.declareArgument("failpenalties", "Negative rewards for misclassifying resp. positives and negatives", 2, "<pospenalty> <negpenalty>" );
    args.declareArgument("learningrate", "The learning rate", 3, "<numerator> <denominator> <denominator increment>" );
    args.declareArgument("explorationrate", "The exploration rate", 3, "<numerator> <denominator> <denominator increment>" );
    args.declareArgument("paramupdate", "The number of episodes required before updating the learning rate and the exploration rate.", 1, "<num>" );
    args.declareArgument("withoutquitQ", "Take the quit action off.", 0, "" );
    args.declareArgument("maxrbfnumber", "The maximum number of RBF per whyp per action.", 1, "<num>" );
    args.declareArgument("incrementalrewardQ", "Give a reward after each evalation.", 0, "" );
    args.declareArgument("qtable", "Load the GSBNF from a file.", 1, "<file>" );
//    args.declareArgument("hashtable", "Load the Q Hash Table from a file.", 1, "<file>" );
    args.declareArgument("budgeted", "Indicate to take features' cost into account.", 0, "" );
    args.declareArgument("featurecosts", "Read the different costs of the features.", 1, "<file>" );
    args.declareArgument("adaptiveexploration", "Sets the epsilon proportional to the number of evaluations.", 1, "<value>" );
    args.declareArgument("debug", "", 1, "<file>");
    args.declareArgument("bootstrap", "The probability of reinjecting a random misclassified example", 1, "<real>");
    args.declareArgument("mil", "Multiple Instance Learning error output.", 0, "" );
}


//---------------------------------------------------------------------------




// This is the entry point for this application


int main(int argc, const char *argv[])
{
	int steps = 0;
    int ges_failed = 0, ges_succeeded = 0, last_succeeded = 0;
    int totalSteps = 0;
    
	// Initialize the random generator
	srand((unsigned int) time(NULL));
	
	// no need to synchronize with C style stream
	std::ios_base::sync_with_stdio(false);
	
#if STABLE_SORT
	cerr << "WARNING: Stable sort active! It might be slower!!" << endl;
#endif
	
	
	//////////////////////////////////////////////////////////////////////////
	// Standard arguments
	nor_utils::Args args;
	
	//////////////////////////////////////////////////////////////////////////
	// Define basic options
	setBasicOptions(args);
	
	
	//////////////////////////////////////////////////////////////////////////
	// Shows the list of available learners
	string learnersComment = "Available learners are:";
	
	vector<string> learnersList;
	BaseLearner::RegisteredLearners().getList(learnersList);
	vector<string>::const_iterator it;
	for (it = learnersList.begin(); it != learnersList.end(); ++it)
	{
		learnersComment += "\n ** " + *it;
		// defaultLearner is defined in Defaults.h
		if ( *it == defaultLearner )
			learnersComment += " (DEFAULT)";
	}
	
	args.declareArgument("learnertype", "Change the type of weak learner. " + learnersComment, 1, "<learner>");
	
	//////////////////////////////////////////////////////////////////////////
	//// Declare arguments that belongs to all weak learners
	BaseLearner::declareBaseArguments(args);
	
	////////////////////////////////////////////////////////////////////////////
	//// Weak learners (and input data) arguments
	for (it = learnersList.begin(); it != learnersList.end(); ++it)
	{
		args.setGroup(*it + " Options");
		// add weaklearner-specific options
		BaseLearner::RegisteredLearners().getLearner(*it)->declareArguments(args);
	}
	
	//////////////////////////////////////////////////////////////////////////
	//// Declare arguments that belongs to all bandit learner
	GenericBanditAlgorithm::declareBaseArguments(args);
	
	
	//////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////
	
	switch ( args.readArguments(argc, argv) )
	{
		case nor_utils::AOT_NO_ARGUMENTS:
			showBase();
			break;
			
		case nor_utils::AOT_UNKOWN_ARGUMENT:
			exit(1);
			break;
			
		case nor_utils::AOT_INCORRECT_VALUES_NUMBER:
			exit(1);
			break;
			
		case nor_utils::AOT_OK:
			break;
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////
	
	if ( args.hasArgument("help") )
		showHelp(args, learnersList);
	if ( args.hasArgument("static") )
		showStaticConfig();
	
	//////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////
	
	if ( args.hasArgument("h") )
		showOptionalHelp(args);
	
	//////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////
	
	int verbose = 1;
	
	if ( args.hasArgument("verbose") )
		args.getValue("verbose", 0, verbose);
	
	//////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////
	
	// defines the seed
	if (args.hasArgument("seed"))
	{
		unsigned int seed = args.getValue<unsigned int>("seed", 0);
		srand(seed);
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////
	
	// Console Input Processing
	if (verbose>5)
	{
		char *debugFile = "debug.txt";
		DebugInit("debug.txt", "+", false);
	}
	
	
	int evalTestIteration=0;
	int episodeNumber = 0;
	
	if (args.hasArgument("episodes"))
	{
		episodeNumber = args.getValue<int>("episodes", 0);
		evalTestIteration = args.getValue<int>("episodes", 1);
	}
	else {
		cout << "No episode argiment!!!!" << endl;
		exit(-1);
	}
	
	string logDirContinous="log";
	if (args.hasArgument("logdir"))
	{
		logDirContinous = args.getValue<string>("logdir", 0);
	}
    
    // only UNIX
    string command = "if [ ! -d \"" + logDirContinous + "\" ]; then mkdir \"" + logDirContinous + "\" ; fi";
    system(command.c_str());

    
    string qTablesDir="qtables";
	if (args.hasArgument("qdir"))
	{
		qTablesDir = args.getValue<string>("qdir", 0);
	}
    
    // only UNIX
    command = "if [ ! -d \"" + qTablesDir + "\" ]; then mkdir \"" + qTablesDir + "\" ; fi";
    system(command.c_str());

	DataReader* datahandler = new DataReader( args, verbose );
	datahandler->setCurrentDataToTrain();
	
    double epsNumerator = 1.;
	double epsDivisor = 4.0;
    double epsIncrement = 0.1;
    
    double qRateNumerator = 0.2 ;
	double qRateDivisor = 1.;
    double qRateIncrement = 1;
    
    
    if (args.hasArgument("learningrate"))
	{
		qRateNumerator = args.getValue<double>("learningrate", 0);
		qRateDivisor = args.getValue<double>("learningrate", 1);
        qRateIncrement = args.getValue<double>("learningrate", 2);
	}
    
    if (args.hasArgument("explorationrate"))
	{
		epsNumerator = args.getValue<double>("explorationrate", 0);
		epsDivisor = args.getValue<double>("explorationrate", 1);
		epsIncrement = args.getValue<double>("explorationrate", 2);
	}
    
    double adaptiveEpsilon = 0;
    if (args.hasArgument("adaptiveexploration"))
	{
		adaptiveEpsilon = args.getValue<double>("adaptiveexploration", 0);
    }

    
    
    double currentEpsilon = epsNumerator / epsDivisor;
    double currentAlpha = qRateNumerator / qRateDivisor;
    
    double lambdaParam = 0.95;
    
    int paramUpdate = 10000;
    if (args.hasArgument("paramupdate"))
	{
		paramUpdate = args.getValue<int>("paramupdate", 0);
    }
    
    if ( datahandler->getClassNumber() <= 2 )
    {
        cout  << "---[ Binary classification ]---" << endl << endl;
    }
    else
    {
        cout << endl << "---[ Multi-class classification ]---" << endl << endl;
    }
    
    int numClasses = datahandler->getClassNumber();
//    if (numClasses == 2) --numClasses;
    
    AdaBoostMDPClassifierContinous* classifierContinous = new AdaBoostMDPClassifierContinous(args, verbose, datahandler, numClasses, 3);
    CRewardFunction *rewardFunctionContinous = classifierContinous;
    
    classifierContinous->setProportionalAlphaNormalization(true);
    
    // Create the agent in our environmentModel.
    CAgent *agentContinous = new CAgent(classifierContinous);
    
    // Add all possible Actions to the agent
    // skip
    agentContinous->addAction(new CAdaBoostAction(0));
    // classify
    agentContinous->addAction(new CAdaBoostAction(1));
    
    if (!args.hasArgument("withoutquitQ"))
    {
        // jump to the end
        agentContinous->addAction(new CAdaBoostAction(2));
    }
    
    CStateModifier* discState = NULL;
    // simple discretized state space
    //CStateModifier* discState = classifierContinous->getStateSpace();
    int featnum = 9;
    if ( args.hasArgument("numoffeat") )
        featnum = args.getValue<int>("numoffeat", 0);
    
    if ( args.hasArgument("etrace") )
        lambdaParam = args.getValue<double>("etrace", 0);
    
    CAbstractQFunction * qData;
    
    #pragma mark State space selection
    
    // some modifs for Son :)
    int sptype = 6;
    if ( args.hasArgument("statespace") )
    {
        sptype = args.getValue<int>("statespace", 0);
    }
    if (sptype==0) {
        if ( datahandler->getClassNumber() <= 2 )
            discState = classifierContinous->getStateSpace(featnum);
        else
            discState = classifierContinous->getStateSpaceExp(featnum,2.0);
        agentContinous->addStateModifier(discState);
        qData = new CFeatureQFunction(agentContinous->getActions(), discState);
    }
    else if (sptype == 5 ) {
        discState = classifierContinous->getStateSpaceForGSBNFQFunction(featnum);
        agentContinous->addStateModifier(discState);
        qData = new GSBNFBasedQFunction(agentContinous->getActions(), discState);
        
        double initRBFs[] = {1.0,1.0,1.0};
        if ( args.hasArgument("optimistic") )
        {
            assert(args.getNumValues("optimistic") == 3);
            initRBFs[0] = args.getValue<double>("optimistic", 0);
            initRBFs[1] = args.getValue<double>("optimistic", 1);
            initRBFs[2] = args.getValue<double>("optimistic", 2);
        }
        
        vector<double> bias(3);
        if ( args.hasArgument("rbfbias") )
        {
            assert(args.getNumValues("rbfbias") == 3);
            bias[0] = args.getValue<double>("rbfbias", 0);
            bias[1] = args.getValue<double>("rbfbias", 1);
            bias[2] = args.getValue<double>("rbfbias", 2);
        }
        
        if (args.hasArgument("qtable")) {
            cout << "Loading Q-Table..." << endl;
            dynamic_cast<GSBNFBasedQFunction*>( qData )->loadQFunction(args.getValue<string>("qtable", 0));
        }

        dynamic_cast<GSBNFBasedQFunction*>( qData )->setBias(bias);
        
        int addCenter = 1;
        if ( args.hasArgument("noaddcenter") )
            addCenter = 0;
        
        int normalizeRbf = 0;
        if ( args.hasArgument("normrbf") )
            normalizeRbf = 1;
        
        double initSigma = 0.01;
        if ( args.hasArgument("rbfsigma") )
            initSigma = args.getValue<double>("rbfsigma", 0);
        
        int maxtderr = 10;
        if ( args.hasArgument("maxtderr") )
            maxtderr = args.getValue<int>("maxtderr", 0);
        
        double minact = 0.4;
        if ( args.hasArgument("minrbfact") )
            minact = args.getValue<double>("minrbfact", 0);
        
        int maxrbfnumber = 1000;
        if ( args.hasArgument("maxrbfnumber") )
            maxrbfnumber = args.getValue<int>("MaxRBFNumber", 0);
        
        cout << "[+] Meta parameters:" << endl;
        cout << "\t--> Normalized RBF: " << normalizeRbf << endl;
        cout << "\t--> RBF Sigma: " << initSigma << endl;
        
        if (addCenter != 0)
        {
            cout << "\t--> New center addition:" << endl;
            cout << "\t\t--> Max TD error: 1/" << maxtderr << endl;
            cout << "\t\t--> Min RBF activation: " << minact << endl;
        }
        cout << endl;
        
        qData->setParameter("AddCenterOnError", addCenter);
        qData->setParameter("NormalizedRBFs", normalizeRbf);
        qData->setParameter("InitRBFSigma", initSigma);
        qData->setParameter("MaxTDErrorDivFactor", maxtderr);
        qData->setParameter("MinActivation", minact);
        qData->setParameter("QLearningRate", currentAlpha);
        qData->setParameter("MaxRBFNumber", maxrbfnumber);
        
        dynamic_cast<GSBNFBasedQFunction*>( qData )->uniformInit(initRBFs);
        
        dynamic_cast<GSBNFBasedQFunction*>( qData )->setMuAlpha(1) ;
        dynamic_cast<GSBNFBasedQFunction*>( qData )->setMuMean(0.00) ;
        dynamic_cast<GSBNFBasedQFunction*>( qData )->setMuSigma(0.00) ;
    }
    else if (sptype == 6) {
        discState = classifierContinous->getStateSpaceForGSBNFQFunction(featnum);
        agentContinous->addStateModifier(discState);
        qData = new HashTable(agentContinous->getActions(), discState, classifierContinous, datahandler->getClassNumber());

        dynamic_cast<HashTable*>(qData)->setScoreResolution(featnum);
        
        if (args.hasArgument("qtable")) {
            cout << "Loading Q Hash Table from : " << args.getValue<string>("qtable", 0) << endl;
            dynamic_cast<HashTable*>( qData )->loadActionValueTable(args.getValue<string>("qtable", 0));
        }

    }
    else {
        cout << "unkown statespcae" << endl;
        exit(1);
    }
    
    
//    else {
//        
//        cout << "No state space resresantion is given. Use --statespace" << endl;
////        discState = classifierContinous->getStateSpace(featnum);
//        exit(1);
//    }
    
    cout << "\t--> Learning rate:" << endl;
    cout << "\t\t--> Numerator: " << qRateNumerator << endl;
    cout << "\t\t--> Divisor: " << qRateDivisor << endl;
    cout << "\t\t--> Increment of the divisor: " << qRateIncrement << endl;
    
    cout << "\t--> Exploration rate:" << endl;
    cout << "\t\t--> Numerator: " << epsNumerator << endl;
    cout << "\t\t--> Divisor: " << epsDivisor << endl;
    cout << "\t\t--> Increment of the divisor: " << epsDivisor << endl;
    
    cout << "\t--> Update frequency: " << paramUpdate << endl;

    cout << "\t--> Reward type: " << args.getValue<string>("succrewartdtype", 0) << endl;

    
    CTDLearner *qFunctionLearner = new CQLearner(classifierContinous, qData);
//    CSarsaLearner *qFunctionLearner = new CSarsaLearner(rewardFunctionContinous, qData, agentContinous);
    
    // Create the Controller for the agent from the QFunction. We will use a EpsilonGreedy-Policy for exploration.
    CAgentController *policy = new CQStochasticPolicy(agentContinous->getActions(), new CEpsilonGreedyDistribution(currentEpsilon), qData);
    
    
    // Set some options of the Etraces which are not default
    qFunctionLearner->setParameter("ReplacingETraces", 1.0);
    qFunctionLearner->setParameter("Lambda", lambdaParam);
    qFunctionLearner->setParameter("DiscountFactor", 1.0);
    
    qFunctionLearner->setParameter("QLearningRate", currentAlpha);
    qData->setParameter("QLearningRate", currentAlpha);
    
    // Add the learner to the agent listener list, so he can learn from the agent's steps.
    agentContinous->addSemiMDPListener(qFunctionLearner);
    agentContinous->setController(policy);
    
    
    // disable automatic logging of the current episode from the agent
    agentContinous->setLogEpisode(false);
    
    int steps2 = 0;
    int usedClassifierNumber=0;
    int max_Steps = 100000;
    double adaboostTrainPerf = 0., adaboostValidPerf = 0., adaboostTestPerf = 0.;
    
    cout << "[+] Computing Adaboost performance..." << flush;
    classifierContinous->setCurrentDataToTrain();
    adaboostTrainPerf = classifierContinous->getAdaboostPerfOnCurrentDataset();
    classifierContinous->setCurrentDataToTest();
    adaboostValidPerf = classifierContinous->getAdaboostPerfOnCurrentDataset();
    
    if (classifierContinous->setCurrentDataToTest2())
        adaboostTestPerf = classifierContinous->getAdaboostPerfOnCurrentDataset();
    
    classifierContinous->setCurrentDataToTrain();
    cout << " done!" << endl;
    if (args.hasArgument("testmdp"))
    {
        
        if (sptype < 5) {
            cout << "Error: use sptype 5 with --testmdp" << endl;
            exit(1);
        }
        
        agentContinous->removeSemiMDPListener(qFunctionLearner);
        
        CAgentController* greedypolicy = new CQGreedyPolicy(agentContinous->getActions(), qData);
        agentContinous->setController(greedypolicy);
        
        if (sptype == 5)
            dynamic_cast<GSBNFBasedQFunction*>( qData )->loadQFunction(args.getValue<string>("testmdp", 0));
        else if (sptype == 6)
            dynamic_cast<HashTable*>( qData )->loadActionValueTable(args.getValue<string>("testmdp", 0));
        
        classifierContinous->setCurrentDataToTrain();
        AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinous> evalTrain( agentContinous, rewardFunctionContinous );
        BinaryResultStruct bres;
        bres.iterNumber=0;
        bres.adaboostPerf = adaboostTrainPerf;
        
        string logFileName = args.getValue<string>("testmdp", 1);
        evalTrain.classficationPerformance(bres,logFileName, true);
        
        classifierContinous->setCurrentDataToTest();
        AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinous> evalTest( agentContinous, rewardFunctionContinous );
        
        bres.adaboostPerf = adaboostTestPerf;
        //            bres.iterNumber=0;
        string logFileName2 = args.getValue<string>("testmdp", 2);
        
        evalTest.classficationPerformance(bres,logFileName2, true);
        
        cout << "******** Overall Test err by MDP: " << bres.err << "(" << adaboostTestPerf << ")" << endl;
        cout << "******** Average Test classifier used: " << bres.usedClassifierAvg << endl;
        cout << "******** Sum of rewards on Test: " << bres.avgReward << endl;
        
        cout << endl << "full" << setw(10) << "prop" << setw(10) << "acc" << setw(10) << "eval" << setw(10) << "rwd" << setw(10) << "cost" ;
        
        if (datahandler->isMILsetup()) {
            cout  << setw(10) << "mil";
        }
        
        cout  << setprecision(4) <<  endl ;

        cout << 100*(1 - bres.adaboostPerf) << setw(10)  <<  100*(1 - bres.itError) << setw(10) << 100*(1 - bres.err) << setw(10) << bres.usedClassifierAvg << setw(10) << bres.avgReward << setw(10) << bres.classificationCost;
        
        if (datahandler->isMILsetup()) {
            cout << setw(10) << bres.milError;
        }
        
        cout << endl;
        
        delete datahandler;
        delete classifierContinous;
        delete agentContinous;
        delete qData;
        delete qFunctionLearner;
        delete policy;

        exit(0);
    }

    if (args.hasArgument("deeparff"))
    {
        
        if (sptype < 5) {
            cout << "Error: use sptype 5 with --deeparff" << endl;
            exit(1);
        }
        
        agentContinous->removeSemiMDPListener(qFunctionLearner);
        
        CAgentController* greedypolicy = new CQGreedyPolicy(agentContinous->getActions(), qData);
        agentContinous->setController(greedypolicy);
        
        if (sptype == 5)
            dynamic_cast<GSBNFBasedQFunction*>( qData )->loadQFunction(args.getValue<string>("deeparff", 0));
        else if (sptype == 6)
            dynamic_cast<HashTable*>( qData )->loadActionValueTable(args.getValue<string>("deeparff", 0));

        int mode = args.getValue<int>("deeparff", 3);
        
        classifierContinous->setCurrentDataToTrain();
        AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinous> evalTrain( agentContinous, rewardFunctionContinous );
        string logFileName = args.getValue<string>("deeparff", 1);
        evalTrain.outputDeepArff(logFileName, mode);
        
        
        classifierContinous->setCurrentDataToTest();
        AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinous> evalTest( agentContinous, rewardFunctionContinous );
        string logFileName2 = args.getValue<string>("deeparff", 2);
        evalTest.outputDeepArff(logFileName2, mode);
        
        
        delete datahandler;
        delete classifierContinous;
        delete agentContinous;
        delete qData;
        delete qFunctionLearner;
        delete policy;
        
        exit(0);
    }

    cout << "Train: " << adaboostTrainPerf << "\t Valid: " << adaboostValidPerf;
    
    if (adaboostTestPerf != 0) {
        cout << "\t Test: " << adaboostTestPerf;
    }
    
    cout << endl;
    
    cout << "---------------------------------" << endl;
    double bestError=numeric_limits<double>::max(), bestWhypNumber=0.;
    int bestEpNumber = 0;
    
    classifierContinous->outHeader();
    
    //TMP
//    currentEpsilon = 0.97;
//    policy->setParameter("EpsilonGreedy", currentEpsilon);
    
    #pragma mark Main loop
    // Learn for 500 Episodes
    for (int i = 0; i < episodeNumber; i++)
    {
//        currentAlpha = 1./(i + 1);
//        qFunctionLearner->setParameter("QLearningRate", currentAlpha );
//        qData->setParameter("QLearningRate", currentAlpha);

        agentContinous->startNewEpisode();
        classifierContinous->setRandomizedInstance();
        steps2 = agentContinous->doControllerEpisode(1, max_Steps);
        
        
        usedClassifierNumber += classifierContinous->getUsedClassifierNumber();
        
        bool clRes = classifierContinous->classifyCorrectly();
        if ( clRes ) {
            ges_succeeded++;
        }
        else {
            ges_failed++;
        }
        
        if (((i%1000)==0) && (i>2))
        {
            cout << "Episode number:  " << SEP  << i << endl;
            cout << "Current error:   " << SEP << (((double)ges_failed / ((double)(ges_succeeded+ges_failed))) * 100.0) << endl;;
            cout << "Used classifier: " << SEP << ((double)usedClassifierNumber / 1000.0) << endl;
            cout << "Current alpha:   "  << SEP  << currentAlpha << endl;
            cout << "Current epsilon: "  << SEP  << policy->getParameter("EpsilonGreedy") << endl;
            cout << "---------------------------------" << endl;
            usedClassifierNumber = 0;
        }
        
        
        if (((i%paramUpdate)==0) && (i>2))
        {
            epsDivisor += epsIncrement;
            currentEpsilon =  epsNumerator / epsDivisor;
            policy->setParameter("EpsilonGreedy", currentEpsilon);
            //policy->setParameter("SoftMaxBeta", currentEpsilon);
            
        }
        if (((i%paramUpdate)==0) && (i>2))
        {
            qRateDivisor += qRateIncrement;
            currentAlpha = qRateNumerator / qRateDivisor;
            qFunctionLearner->setParameter("QLearningRate", currentAlpha);
            qData->setParameter("QLearningRate", currentAlpha);
        }
        
#pragma mark evalation
        
        if (((i%evalTestIteration)==0) && (i>2))
        {            
            agentContinous->removeSemiMDPListener(qFunctionLearner);
            
            // set the policy to be greedy
            // Create the learners controller from the Q-Function, we use a SoftMaxPolicy
            CAgentController* greedypolicy = new CQGreedyPolicy(agentContinous->getActions(), qData);
            
            // set the policy as controller of the agent
            agentContinous->setController(greedypolicy);
            
            // TRAIN stats
            classifierContinous->setCurrentDataToTrain();
            AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinous> evalTrain( agentContinous, rewardFunctionContinous );
            
            BinaryResultStruct bres;
            bres.adaboostPerf = adaboostTrainPerf;
            bres.iterNumber=i;
            
            evalTrain.classficationPerformance(bres, "");
            
//            if (((i%10000)==0) && (i>2))
//            {
//                policy->setParameter("EpsilonGreedy", 0.5/bres.usedClassifierAvg);
//                
//            }
            
            if (adaptiveEpsilon > 0 )
                if (bres.usedClassifierAvg != 0)
                    policy->setParameter("EpsilonGreedy", adaptiveEpsilon/bres.usedClassifierAvg);
                else
                    policy->setParameter("EpsilonGreedy", adaptiveEpsilon);


            cout << "[+] Training set results: " << endl;
            cout << "--> Overall error by MDP: " << bres.err << " (" << classifierContinous->getIterationError((int)bres.usedClassifierAvg) << ")" <<  " (" << adaboostTrainPerf << ")" << endl;
            cout << "--> Average classifier used: " << bres.usedClassifierAvg << endl;
            cout << "--> Sum of rewards: " << bres.avgReward << endl << endl;
            
            classifierContinous->outPutStatistic( bres );
            
            
            // VALID stats
            
            classifierContinous->setCurrentDataToTest();
            AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinous> evalValid( agentContinous, rewardFunctionContinous );
            
            bres.adaboostPerf = adaboostValidPerf;
            bres.iterNumber=i;
            
            string logFileName;
            if (!logDirContinous.empty()) {
                char logfname[4096];
                sprintf( logfname, "%s/classValid_%d.txt", logDirContinous.c_str(), i );
                logFileName = string(logfname);
            }
            
            evalValid.classficationPerformance(bres, logFileName);
            
            if (bres.err < bestError) {
                bestEpNumber = i;
                bestError = bres.err;
                bestWhypNumber = bres.usedClassifierAvg;
            }
            
            cout << "[+] Validation set results: " << endl;
            cout << "--> Overall error by MDP: " << bres.err << " (" << classifierContinous->getIterationError((int)bres.usedClassifierAvg) << ")" << " (" << adaboostValidPerf << ")" << endl;
            cout << "--> Average classifier used: " << bres.usedClassifierAvg << endl;
            cout << "--> Sum of rewards: " << bres.avgReward << endl << endl;
            
            classifierContinous->outPutStatistic( bres );
            
            cout << "----> Best error so far ( " << bestEpNumber << " ) : " << bestError << endl << "----> Num of whyp used : " << bestWhypNumber << endl << endl;
            
            // TEST stats
            
            if (classifierContinous->setCurrentDataToTest2() )
            {
                AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinous> evalTest( agentContinous, rewardFunctionContinous );
                
                bres.adaboostPerf = adaboostTestPerf;
                
                if (!logDirContinous.empty()) {
                    char logfname[4096];
                    sprintf( logfname, "%s/classTest_%d.txt", logDirContinous.c_str(), i );
                    logFileName = string(logfname);
                }
                
                evalTest.classficationPerformance(bres,logFileName);
                
                
                cout << "--> Overall Test error by MDP: " << bres.err << " (" << classifierContinous->getIterationError((int)bres.usedClassifierAvg) << ")" << " (" << adaboostTestPerf << ")" << endl;
                cout << "--> Average Test classifier used: " << bres.usedClassifierAvg << endl;
                cout << "--> Sum of rewards on Test: " << bres.avgReward << endl << endl;
                
                classifierContinous->outPutStatistic( bres );
            }
            
            cout << "---------------------------------" << endl;
            
            if (sptype == 0) {
                std::stringstream ss;
                ss << qTablesDir << "/QTable_" << i << ".dta";
                FILE *qTableFile2 = fopen(ss.str().c_str(), "w");
                dynamic_cast<CFeatureQFunction*>(qData)->saveFeatureActionValueTable(qTableFile2);
                fclose(qTableFile2);
            }
            
            if (sptype == 5) {
                std::stringstream ss;
                ss << qTablesDir << "/QTable_" << i << ".dta";
                FILE *qTableFile2 = fopen(ss.str().c_str(), "w");
                dynamic_cast<GSBNFBasedQFunction*>(qData)->saveActionValueTable(qTableFile2);
                fclose(qTableFile2);
            }
            
            if (sptype == 6) {
                std::stringstream ss;
                ss << qTablesDir << "/QTable_" << i << ".dta";
                FILE* qTableFile2 = fopen(ss.str().c_str(), "w");
                dynamic_cast<HashTable*>(qData)->saveActionValueTable(qTableFile2);
                fclose(qTableFile2);
                
                string lastQTableFileName = "last_qtable.dta";
                FILE* lastQTable = fopen(lastQTableFileName.c_str(), "w");
                dynamic_cast<HashTable*>(qData)->saveActionValueTable(lastQTable);
                fclose(lastQTable);
            }

            agentContinous->setController(policy);
            agentContinous->addSemiMDPListener(qFunctionLearner);
            classifierContinous->setCurrentDataToTrain();
        }
    }

    delete datahandler;
    delete classifierContinous;
    delete agentContinous;
    delete qData;
    delete qFunctionLearner;
    delete policy;

}

