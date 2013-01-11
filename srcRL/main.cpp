//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// general includes

#include <time.h>

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
#include "ctorchvfunction.h"
#include "ccontinuousactions.h"
#include "MLP.h"
#include "GradientMachine.h"
#include "LogRBF.h"
#include "Linear.h"
#include "Tanh.h"
#include "RBFBasedQFunction.h"

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// for multiboost
#include "Defaults.h"
#include "Utils/Args.h"

#include "StrongLearners/GenericStrongLearner.h"
#include "WeakLearners/BaseLearner.h" // To get the list of the registered weak learners

#include "IO/Serialization.h" // for unserialization
#include "Bandits/GenericBanditAlgorithm.h"
//#include "AdaBoostMDPClassifier.h"
#include "AdaBoostMDPClassifierAdv.h"
//#include "AdaBoostMDPClassifierContinous.h"
#include "AdaBoostMDPClassifierDiscrete.h"
#include "AdaBoostMDPClassifierContinousBinary.h"
#include "AdaBoostMDPClassifierContinousMultiClass.h"
//#include "AdaBoostMDPClassifierSubsetSelectorBinary.h"

using namespace std;
using namespace MultiBoost;
using namespace Torch;

#define SEP setw(15)

//#define LOGQTABLE
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

static const char CURRENT_VERSION[] = "1.0.00";


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
	cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
	cout << "---------------------------------------------------------------------------" << endl;
	cout << "Build: " << __DATE__ << " (" << __TIME__ << ") (C) Robert Busa-Fekete, Balazs Kegl, Norman Casagrande 2005-2010" << endl << endl;
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
	cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
	cout << "------------------------ HELP SECTION --------------------------" << endl;
	
	args.printGroup("Parameters");
	
	cout << endl;
	cout << "For specific help options type:" << endl;
	cout << "   --h general: General options" << endl;
	cout << "   --h io: I/O options" << endl;
	cout << "   --h algo: Basic algorithm options" << endl;
	cout << "   --h bandits: Bandit algorithm options" << endl;
	
	cout << endl;
	cout << "For weak learners specific options type:" << endl;
	
	vector<string>::const_iterator it;
	for (it = learnersList.begin(); it != learnersList.end(); ++it)
		cout << "   --h " << *it << endl;
	
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
	
	cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
	cout << "---------------------------------------------------------------------------" << endl;
	
	if (helpType == "general")
		args.printGroup("General Options");
	else if (helpType == "io")
		args.printGroup("I/O Options");
	else if (helpType == "algo")
		args.printGroup("Basic Algorithm Options");
	else if (helpType == "bandits")
		args.printGroup("Bandit Algorithm Options");
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
	
	args.declareArgument("train", "Performs training.", 2, "<dataFile> <nInterations>");
	args.declareArgument("traintestmdp", "Performs training and test at the same time.", 5, "<trainingDataFile> <testDataFile> <nInterations> <shypfile> <outfile>");
    args.declareArgument("traintestmdp", "Performs training and test at the same time.", 6, "<trainingDataFile> <validDataFile> <nInterations> <shypfile> <outfile> <testDataFile>");
    args.declareArgument("testmdp", "Performs test of a previously leant model.", 3, "<qtable> <train log file> <test log file>");
	args.declareArgument("test", "Test the model.", 3, "<dataFile> <numIters> <shypFile>");
	args.declareArgument("test", "Test the model and output the results", 4, "<datafile> <shypFile> <numIters> <outFile>");
	args.declareArgument("cmatrix", "Print the confusion matrix for the given model.", 2, "<dataFile> <shypFile>");
	args.declareArgument("cmatrixfile", "Print the confusion matrix with the class names to a file.", 3, "<dataFile> <shypFile> <outFile>");
	args.declareArgument("posteriors", "Output the posteriors for each class, that is the vector-valued discriminant function for the given dataset and model.", 4, "<dataFile> <shypFile> <outFile> <numIters>");
	args.declareArgument("posteriors", "Output the posteriors for each class, that is the vector-valued discriminant function for the given dataset and model periodically.", 5, "<dataFile> <shypFile> <outFile> <numIters> <period>");
	args.declareArgument("cposteriors", "Output the calibrated posteriors for each class, that is the vector-valued discriminant function for the given dataset and model.", 4, "<dataFile> <shypFile> <outFile> <numIters>");
	
	args.declareArgument("likelihood", "Output the likelihoof of data for each iteration, that is the vector-valued discriminant function for the given dataset and model.", 4, "<dataFile> <shypFile> <outFile> <numIters>");
	
	args.declareArgument("encode", "Save the coefficient vector of boosting individually on each point using ParasiteLearner", 6, "<inputDataFile> <autoassociativeDataFile> <outputDataFile> <nIterations> <poolFile> <nBaseLearners>");
	args.declareArgument("roc", "Print out the ROC curve (it calculate the ROC curve for the first class)", 4, "<dataFile> <shypFile> <outFile> <numIters>" );
	
	args.declareArgument("ssfeatures", "Print matrix data for SingleStump-Based weak learners (if numIters=0 it means all of them).", 4, "<dataFile> <shypFile> <outFile> <numIters>");
	
	args.declareArgument( "fileformat", "Defines the type of intput file. Available types are:\n"
						 "* simple: each line has attributes separated by whitespace and class at the end (DEFAULT!)\n"
						 "* arff: arff filetype. The header file can be specified using --arffheader option\n"
						 "* arffbzip: bziped arff filetype. The header file can be specified using --arffheader option\n"
						 "* svmlight: \n"
						 "(Example: --fileformat simple)",
                         1, "<fileFormat>" );
	
	args.declareArgument("headerfile", "The filename of the header file (SVMLight).", 1, "header.txt");
	
	args.declareArgument("constant", "Check constant learner in each iteration.", 0, "");
	args.declareArgument("timelimit", "Time limit in minutes", 1, "<minutes>" );
	args.declareArgument("stronglearner", "Strong learner. Available strong learners:\n"
						 "AdaBoost (default)\n"
						 "BrownBoost\n", 1, "<stronglearner>" );
	
	args.declareArgument("slowresumeprocess", "Compute the results in each iteration (slow resume)\n"
						 "Compute only the data of the last iteration (fast resume, default)\n", 0, "" );
	args.declareArgument("weights", "Outputs the weights of instances at the end of the learning process", 1, "<filename>" );
	args.declareArgument("Cn", "Resampling size for FilterBoost (default=300)", 1, "<val>" );
	//// ignored for the moment!
	//args.declareArgument("arffheader", "Specify the arff header.", 1, "<arffHeaderFile>");
	
	//////////////////////////////////////////////////////////////////////////
	// Options
	
	args.setGroup("I/O Options");
	
	/////////////////////////////////////////////
	// these are valid only for .txt input!
	// they might be removed!
	args.declareArgument("d", "The separation characters between the fields (default: whitespaces).\nExample: -d \"\\t,.-\"\nNote: new-line is always included!", 1, "<separators>");
	args.declareArgument("classend", "The class is the last column instead of the first (or second if -examplelabel is active).");
	args.declareArgument("examplename", "The data file has an additional column (the very first) which contains the 'name' of the example.");
	
	/////////////////////////////////////////////
	
	args.setGroup("Basic Algorithm Options");
	args.declareArgument("weightpolicy", "Specify the type of weight initialization. The user specified weights (if available) are used inside the policy which can be:\n"
						 "* sharepoints Share the weight equally among data points and between positiv and negative labels (DEFAULT)\n"
						 "* sharelabels Share the weight equally among data points\n"
						 "* proportional Share the weights freely", 1, "<weightType>");
	
	
	args.setGroup("General Options");
	
	args.declareArgument("verbose", "Set the verbose level 0, 1 or 2 (0=no messages, 1=default, 2=all messages).", 1, "<val>");
	args.declareArgument("outputinfo", "Output informations on the algorithm performances during training, on file <filename>.", 1, "<filename>");
	args.declareArgument("seed", "Defines the seed for the random operations.", 1, "<seedval>");
	
	//////////////////////////////////////////////////////////////////////////
	// Options for TL tool
	args.setGroup("RL options");
	
	args.declareArgument("gridworldfilename", "The naem of gridwold description filename", 1, "<val>");
	args.declareArgument("episodes", "The number of episodes", 2, "<episod> <testiter>");
	args.declareArgument("rewards", "success, class, skip", 3, "<succ> <class> <skip>");
	args.declareArgument("logdir", "Dir of log", 1, "<dir>");
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
	//////////////////////////////////////////////////////////////////////////////////////////
	string gridworldFileName = "Gridworld_10x10.txt";
	
	// Console Input Processing
	if (verbose>5)
	{
		char *debugFile = "debug.txt";
		DebugInit("debug.txt", "+", false);
	}
	
	if (args.hasArgument("gridworldfilename"))
	{
		args.getValue("gridworldfilename", 0, gridworldFileName);
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
	
	string logDirContinous="";
	if (args.hasArgument("logdir"))
	{
		logDirContinous = args.getValue<string>("logdir", 0);
	}
	
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
    
    
    double currentEpsilon = epsNumerator / epsDivisor;
    double currentAlpha = qRateNumerator / qRateDivisor;
    
    double lambdaParam = 0.95;
    
    int paramUpdate = 10000;
    if (args.hasArgument("paramupdate"))
	{
		paramUpdate = args.getValue<double>("paramupdate", 0);
    }
    
    AdaBoostMDPClassifierContinous* classifierContinous;
    if ( datahandler->getClassNumber() <= 2 )
    {
        cout << endl << "---[ Binary classification ]---" << endl << endl;
        classifierContinous = new AdaBoostMDPClassifierContinousBinary(args, verbose, datahandler );
    }
    else
    {
        cout << endl << "---[ Multi-class classification ]---" << endl << endl;
        classifierContinous = new AdaBoostMDPClassifierContinousMH(args, verbose, datahandler, datahandler->getClassNumber() );
    }
    
    CRewardFunction *rewardFunctionContinous = classifierContinous;
    
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
    int featnum = 11;
    if ( args.hasArgument("numoffeat") )
        featnum = args.getValue<int>("numoffeat", 0);
    
    if ( args.hasArgument("etrace") )
        lambdaParam = args.getValue<double>("etrace", 0);
    
    CAbstractQFunction * qData;
    
    int sptype = -1;
    if ( args.hasArgument("statespace") )
    {
        sptype = args.getValue<int>("statespace", 0);
        
        if (sptype==0) {
            if ( datahandler->getClassNumber() <= 2 )
                discState = classifierContinous->getStateSpace(featnum);
            else
                discState = dynamic_cast<AdaBoostMDPClassifierContinousMH*>(classifierContinous)->getStateSpaceExp(featnum,2.0);
            agentContinous->addStateModifier(discState);
            qData = new CFeatureQFunction(agentContinous->getActions(), discState);
        }
        else if (sptype ==5 ) {
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

            cout << "\t--> Learning rate:" << endl;
            cout << "\t\t--> Numerator: " << qRateNumerator << endl;
            cout << "\t\t--> Divisor: " << qRateDivisor << endl;
            cout << "\t\t--> Increment of the divisor: " << qRateIncrement << endl;
            
            cout << "\t--> Exploration rate:" << endl;
            cout << "\t\t--> Numerator: " << epsNumerator << endl;
            cout << "\t\t--> Divisor: " << epsDivisor << endl;
            cout << "\t\t--> Increment of the divisor: " << epsDivisor << endl;
            
            cout << "\t--> Update frequency: " << paramUpdate << endl;
            
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
        else {
            cout << "unkown statespcae" << endl;
        }
        
    } else {
        cout << "No state space resresantion is given, the default is used" << endl;
        discState = classifierContinous->getStateSpace(featnum);
    }
    
    
    CTDLearner *qFunctionLearner = new CQLearner(classifierContinous, qData);
    //		CSarsaLearner *qFunctionLearner = new CSarsaLearner(rewardFunctionContinous, qData, agentContinous);
    
    //gradient stuff !!!
    //        CDiscreteResidual* residualFunction = new CDiscreteResidual(0.95);
    //        CConstantBetaCalculator* betaCalculator = new CConstantBetaCalculator(1);
    //        //        CVariableBetaCalculator * betaCalculator = new CVariableBetaCalculator(0.1, 0.99) ; //mu and maxBeta
    //        CResidualBetaFunction* residualGradient = new CResidualBetaFunction(betaCalculator, residualFunction);
    
    
    //        CTDGradientLearner *qFunctionLearner = new CTDGradientLearner(rewardFunctionContinous, qData, agentContinous, residualFunction, residualGradient);
    //        CTDResidualLearner *qFunctionLearner = new CTDResidualLearner(rewardFunctionContinous, dynamic_cast<CGradientQFunction*>(qData), agentContinous, residualFunction, residualGradient, betaCalculator);
    
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
    double ovaccTrain, ovaccValid, ovaccTest;
    
    cout << "[+] Computing Adaboost performance..." << flush;
    classifierContinous->setCurrentDataToTrain();
    ovaccTrain = classifierContinous->getAccuracyOnCurrentDataSet();
    classifierContinous->setCurrentDataToTest();
    ovaccValid = classifierContinous->getAccuracyOnCurrentDataSet();
    
    if (classifierContinous->setCurrentDataToTest2())
        ovaccTest = classifierContinous->getAccuracyOnCurrentDataSet();
    
    classifierContinous->setCurrentDataToTrain();
    cout << " done!" << endl;
    if (args.hasArgument("testmdp"))
    {
        //FIXME: works only with binary classification
        agentContinous->removeSemiMDPListener(qFunctionLearner);
        
        CAgentController* greedypolicy = new CQGreedyPolicy(agentContinous->getActions(), qData);
        agentContinous->setController(greedypolicy);
        
        dynamic_cast<GSBNFBasedQFunction*>( qData )->loadQFunction(args.getValue<string>("testmdp", 0));
        
        classifierContinous->setCurrentDataToTrain();
        AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinousBinary> evalTrain( agentContinous, rewardFunctionContinous );
        BinaryResultStruct bres;
        bres.iterNumber=0;
        bres.origAcc = ovaccTrain;
        
        string logFileName = args.getValue<string>("testmdp", 1);
        evalTrain.classficationAccruacy(bres,logFileName, true);
        
        classifierContinous->setCurrentDataToTest();
        AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinousBinary> evalTest( agentContinous, rewardFunctionContinous );
        
        bres.origAcc = ovaccTest;
        //            bres.iterNumber=0;
        string logFileName2 = args.getValue<string>("testmdp", 2);
        
        evalTest.classficationAccruacy(bres,logFileName2, true);
        
        cout << "******** Overall Test accuracy by MDP: " << bres.acc << "(" << ovaccTest << ")" << endl;
        cout << "******** Average Test classifier used: " << bres.usedClassifierAvg << endl;
        cout << "******** Sum of rewards on Test: " << bres.avgReward << endl;
        
        exit(0);
    }
    
    cout << "Train: " << ovaccTrain << "\t Valid: " << ovaccValid;
    
    if (ovaccTest != 0) {
        cout << "\t Test: " << ovaccTest;
    }
    
    cout << endl;
    
    cout << "---------------------------------" << endl;
    double bestAcc=0., bestWhypNumber=0.;
    int bestEpNumber = 0;
    
    classifierContinous->outHeader();
    
    // Learn for 500 Episodes
    for (int i = 0; i < episodeNumber; i++)
    {
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
            cout << "Current Accuracy:" << SEP << (((float)ges_succeeded / ((float)(ges_succeeded+ges_failed))) * 100.0) << endl;;
            cout << "Used Classifier: " << SEP << ((float)usedClassifierNumber / 1000.0) << endl;
            cout << "Current alpha:   "  << SEP  << currentAlpha << endl;
            cout << "Current Epsilon: "  << SEP  << currentEpsilon << endl;
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
        
        if (((i%evalTestIteration)==0) && (i>2))
        {
#pragma mark binary evaluation
            if ( datahandler->getClassNumber() <= 2 ) {
                agentContinous->removeSemiMDPListener(qFunctionLearner);
                
                // set the policy to be greedy
                // Create the learners controller from the Q-Function, we use a SoftMaxPolicy
                CAgentController* greedypolicy = new CQGreedyPolicy(agentContinous->getActions(), qData);
                
                // set the policy as controller of the agent
                agentContinous->setController(greedypolicy);
                
                // TRAIN stats
                classifierContinous->setCurrentDataToTrain();
                //AdaBoostMDPClassifierContinousBinaryEvaluator evalTrain( agentContinous, rewardFunctionContinous );
                AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinousBinary> evalTrain( agentContinous, rewardFunctionContinous );
                
                BinaryResultStruct bres;
                bres.origAcc = ovaccTrain;
                bres.iterNumber=i;
                
                evalTrain.classficationAccruacy(bres, "");
                
                cout << "[+] Training set results: " << endl;
                cout << "--> Overall accuracy by MDP: " << bres.acc << " (" << ovaccTrain << ")" << endl;
                cout << "--> Average classifier used: " << bres.usedClassifierAvg << endl;
                cout << "--> Sum of rewards: " << bres.avgReward << endl << endl;
                
                //                cout << "----> Best accuracy so far : " << bestAcc << endl << "----> Num of whyp used : " << bestWhypNumber << endl << endl;
                
                classifierContinous->outPutStatistic( bres );
                
                
                // VALID stats
                
                classifierContinous->setCurrentDataToTest();
                //AdaBoostMDPClassifierContinousBinaryEvaluator evalTrain( agentContinous, rewardFunctionContinous );
                AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinousBinary> evalValid( agentContinous, rewardFunctionContinous );
                
                bres.origAcc = ovaccValid;
                bres.iterNumber=i;
                
                string logFileName;
                if (!logDirContinous.empty()) {
                    char logfname[4096];
                    sprintf( logfname, "./%s/classValid_%d.txt", logDirContinous.c_str(), i );
                    logFileName = string(logfname);
                }
                
                evalValid.classficationAccruacy(bres, logFileName);
                
                if (bres.acc > bestAcc) {
                    bestEpNumber = i;
                    bestAcc = bres.acc;
                    bestWhypNumber = bres.usedClassifierAvg;
                }
                
                cout << "[+] Validation set results: " << endl;
                cout << "--> Overall accuracy by MDP: " << bres.acc << " (" << ovaccValid << ")" << endl;
                cout << "--> Average classifier used: " << bres.usedClassifierAvg << endl;
                cout << "--> Sum of rewards: " << bres.avgReward << endl << endl;
                
                //                cout << "----> Best accuracy so far : " << bestAcc << endl << "----> Num of whyp used : " << bestWhypNumber << endl << endl;
                
                classifierContinous->outPutStatistic( bres );
                
                cout << "----> Best accuracy so far ( " << bestEpNumber << " ) : " << bestAcc << endl << "----> Num of whyp used : " << bestWhypNumber << endl << endl;
                
                // TEST stats
                
                if (classifierContinous->setCurrentDataToTest2() )
                {
                    //AdaBoostMDPClassifierContinousBinaryEvaluator evalTest( agentContinous, rewardFunctionContinous );
                    AdaBoostMDPBinaryDiscreteEvaluator<AdaBoostMDPClassifierContinousBinary> evalTest( agentContinous, rewardFunctionContinous );
                    
                    bres.origAcc = ovaccTest;
                    
                    if (!logDirContinous.empty()) {
                        char logfname[4096];
                        sprintf( logfname, "./%s/classTest_%d.txt", logDirContinous.c_str(), i );
                        logFileName = string(logfname);
                    }
                    
                    evalTest.classficationAccruacy(bres,logFileName);
                    
                    
                    //                ss.clear();
                    //                ss << "qtables/ActionTable_" << i << ".dta";
                    //                FILE *actionTableFile2 = fopen(ss.str().c_str(), "w");
                    //                dynamic_cast<RBFBasedQFunctionBinary*>(qData)->saveActionTable(actionTableFile2);
                    //                fclose(actionTableFile2);
                    
                    cout << "--> Overall Test accuracy by MDP: " << bres.acc << " (" << ovaccTest << ")" << endl;
                    cout << "--> Average Test classifier used: " << bres.usedClassifierAvg << endl;
                    cout << "--> Sum of rewards on Test: " << bres.avgReward << endl << endl;
                    
                    classifierContinous->outPutStatistic( bres );
                }
                
                cout << "---------------------------------" << endl;
                
                if (sptype == 5) {
                    std::stringstream ss;
                    ss << "qtables/QTable_" << i << ".dta";
                    FILE *qTableFile2 = fopen(ss.str().c_str(), "w");
                    dynamic_cast<GSBNFBasedQFunction*>(qData)->saveActionValueTable(qTableFile2);
                    fclose(qTableFile2);
                }                
                
                agentContinous->setController(policy);
                agentContinous->addSemiMDPListener(qFunctionLearner);
                classifierContinous->setCurrentDataToTrain();
            }
            else
            {
#pragma mark multiclass evaluation
                char logfname[4096];
                /*
                 sprintf( logfname, "./%s/qfunction_%d.txt", logDirContinous.c_str(), i );
                 FILE *vFuncFileAB = fopen(logfname,"w");
                 qData->saveData(vFuncFileAB);
                 fclose(vFuncFileAB);
                 */
                agentContinous->removeSemiMDPListener(qFunctionLearner);
                
                // Create the learners controller from the Q-Function, we use a SoftMaxPolicy
                CAgentController* greedypolicy = new CQGreedyPolicy(agentContinous->getActions(), qData);
                
                // set the policy as controller of the agent
                agentContinous->setController(greedypolicy);
                
                
                // TRAIN
                classifierContinous->setCurrentDataToTrain();
                AdaBoostMDPClassifierContinousEvaluator evalTrain( agentContinous, rewardFunctionContinous );
                
                double acc, usedclassifierNumber;
                sprintf( logfname, "./%s/classValid_%d.txt", logDirContinous.c_str(), i );
                double sumRew = evalTrain.classficationAccruacy(acc,usedclassifierNumber,logfname);
                
                
                if (sptype == 5) {
                    //save the number of centers per wc per action
                    std::stringstream ss;
                    ss << logDirContinous << "/rbfCenters_" << i << ".dta";
                    FILE* rbfCentersFile = fopen(ss.str().c_str(), "w");
                    vector<int> maxNumCenters = dynamic_cast<GSBNFBasedQFunction*>(qData)->saveCentersNumber(rbfCentersFile);
                    cout << "[+] Max number of RBFs: ";
                    for (int k=0; k < maxNumCenters.size(); ++k) {
                        cout << maxNumCenters[k] << "\t";
                    }
                    cout << endl << endl;
                    fclose(rbfCentersFile);
                }
                
                cout << "******** Overall Train accuracy by MDP: " << acc << " (" << ovaccTrain << ")" << endl;
                cout << "******** Average Train classifier used: " << usedclassifierNumber << endl;
                cout << "******** Sum of rewards on Train: " << sumRew << endl << endl;
                //				cout << "----> Best accuracy so far ( " << bestEpNumber << " ) : " << bestAcc << endl << "----> Num of whyp used : " << bestWhypNumber << endl;
                
                classifierContinous->outPutStatistic(i, ovaccTrain, acc, usedclassifierNumber, sumRew );
                
                
                // TEST
                classifierContinous->setCurrentDataToTest();
                AdaBoostMDPClassifierContinousEvaluator evalTest( agentContinous, rewardFunctionContinous);
                
                
                sprintf( logfname, "./%s/classTest_%d.txt", logDirContinous.c_str(), i );
                sumRew = evalTest.classficationAccruacy(acc,usedclassifierNumber,logfname);
                
                if (acc > bestAcc) {
                    bestAcc = acc;
                    bestWhypNumber = usedclassifierNumber;
                    bestEpNumber = i;
                }
                
                cout << "******** Overall Test accuracy by MDP: " << acc << " (" << ovaccValid << ")" << endl;
                cout << "******** Average Test classifier used: " << usedclassifierNumber << endl;
                cout << "******** Sum of rewards on Test: " << sumRew << endl << endl;
                cout << "----> Best accuracy so far ( " << bestEpNumber << " ) : " << bestAcc << endl
                << "----> Num of whyp used : " << bestWhypNumber << endl << endl;
                classifierContinous->outPutStatistic(i, ovaccValid, acc, usedclassifierNumber, sumRew );
                
                classifierContinous->setCurrentDataToTrain();
                
                /*
                 sprintf( logfname, "./%s/qfunction_%d_2.txt", logDirContinous.c_str(), i );
                 FILE *vFuncFileAB2 = fopen(logfname,"w");
                 qData->saveData(vFuncFileAB2);
                 fclose(vFuncFileAB2);
                 */
                
                agentContinous->addSemiMDPListener(qFunctionLearner);
                agentContinous->setController(policy);
            }
        }
    }

    delete datahandler;
    delete classifierContinous;
    delete agentContinous;
    delete qData;
    delete qFunctionLearner;
    delete policy;

}

