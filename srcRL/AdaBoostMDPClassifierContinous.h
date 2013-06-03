/*
 *  AdaBoostMDPClassifierContinous.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/11/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADABOOST_MDP_CLASS_CONTINOUS_H
#define __ADABOOST_MDP_CLASS_CONTINOUS_H

//////////////////////////////////////////////////////////////////////
// for multiboost
//////////////////////////////////////////////////////////////////////
#include "WeakLearners/BaseLearner.h"
#include "IO/InputData.h"
#include "Utils/Utils.h"
#include "IO/Serialization.h"
#include "IO/OutputInfo.h"
#include "Classifiers/AdaBoostMHClassifier.h"
#include "Classifiers/ExampleResults.h"

//////////////////////////////////////////////////////////////////////
// for RL toolbox
//////////////////////////////////////////////////////////////////////
#include "cenvironmentmodel.h"
#include "crewardfunction.h"
#include "caction.h"
#include "cdiscretizer.h"
#include "cevaluator.h"
#include "cagent.h"
#include "cdiscretizer.h"
#include "cstate.h"
#include "cstatemodifier.h"
#include "clinearfafeaturecalculator.h"
//////////////////////////////////////////////////////////////////////
// general includes
//////////////////////////////////////////////////////////////////////
#include "AdaBoostMDPClassifierAdv.h"

//#include "tbb/parallel_for.h"
//#include "tbb/blocked_range.h"

using namespace std;

namespace MultiBoost {
	////////////////////////////////////////////////////////////////////////////////////////////////	
	enum SuccesRewardModes {
		RT_HAMMING,
		RT_EXP,
        RT_LOGIT,
	};

    //////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////
	struct BinaryResultStruct {
		double adaboostPerf;
		
		double acc;
        double err;
		double TP;
		double TN;
		double usedClassifierAvg;
		double avgReward;
		
        double itError;
        
		int iterNumber;
        double negNumEval;
        
        double classificationCost;
        
        double milError;
	};
    
    typedef vector<int> KeyType;
    typedef map<KeyType, int> KeyIndicesType;

	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////
	class AdaBoostMDPClassifierContinous : public	CEnvironmentModel,	public CRewardFunction
	{
	protected: 
		// 
		int						_verbose;
		int						_classNum;
		int						_classifierNumber; // number of classifier used during the episode
		vector<bool>			_classifierUsed; // store which classifier was used during the process
		vector<int> 			_classifiersOutput;
        
		// rewards
		double					_classificationReward;
        double					_misclassificationReward;
        
		double					_skipReward;
		double					_jumpReward;
		double					_successReward;
		SuccesRewardModes		_succRewardMode;
        
        bool                    _incrementalReward;
        double                  _lastReward;

		DataReader*				_data;
		
		/// internal state variables 
		ExampleResults*			_exampleResult;
		int						_currentClassifier;
		double					_currentSumAlpha;
		
		// for output info
		ofstream				_outputStream;
		
		//!< The arguments defined by the user.		
		const nor_utils::Args&		_args;  
		
		/// calculate the next state based on the action virtual 
		void doNextState(CPrimitiveAction *act);
		
		// this instance will be used in this episode
		int						_currentRandomInstance;
		
		// contain the sum of alphas
		double					_sumAlpha;
        
        bool                    _budgetedClassification;
        vector<AlphaReal>       _featureCosts;
        vector<bool>            _featuresEvaluated;

        string                  _positiveLabelName;
        int                     _positiveLabelIndex;
        
        KeyIndicesType          _keysIndices;
        int                     _currentKeyIndex;
        
        map<vector<int>, int>   _winnersIndices;
        int                     _currentWinnerIndex;
        
        bool                    _proportionalAlphaNorm;
        
        vector<bool>            _lastCorrectClassifications;
        
        ofstream                _debugFileStream;
        
        double                  _bootstrapRate;
                
	public:
		// set randomzed element
		void setCurrentRandomIsntace( int r ) { _currentRandomInstance = r; }		
		void setRandomizedInstance() {
            
            double r = rand()/static_cast<float>(RAND_MAX);
            if (r < _bootstrapRate)
            {
                vector<int> candidates;
                candidates.reserve(_lastCorrectClassifications.size());
    //            memory leaks
                for (int i = 0; i < _lastCorrectClassifications.size(); ++i) {
                    if (_lastCorrectClassifications[i ] == false)
                        candidates.push_back(i);
                }
                _currentRandomInstance = (int) (rand() % candidates.size() );
            }
            else
                _currentRandomInstance = (int) (rand() % _data->getNumExamples() );
            
//            if (_debugFileStream.good()) {
//                _debugFileStream << _currentRandomInstance << " ";
//            }
        }
		
		// getter setter
		int getUsedClassifierNumber() { return _classifierNumber; }
		void setClassificationReward( double r ) { _classificationReward=r; }
		void setSkipReward( double r ) { _skipReward=r; }		
		void setJumpReward( double r ) { _jumpReward=r; }
		void setSuccessReward( double r ) { _successReward=r; }
        
        void setProportionalAlphaNormalization(bool b) {_proportionalAlphaNorm = b;}
		
		int getIterNum() { return _data->getIterationNumber(); };
		int getNumClasses() { return _data->getClassNumber(); };
		int getNumExamples() { return _data->getNumExamples(); }
		void getHistory( vector<bool>& history );
		void getHistory( vector<int>& history );
        
        KeyType getHistoryFromState(int i) {
            KeyIndicesType::const_iterator kIt = _keysIndices.begin();
            for (; kIt != _keysIndices.end(); ++kIt) {
                if (kIt->second == i) {
                    return kIt->first;
                }
            }
            assert(false); //return KeyType();
        }

        vector<int> getWinnersFromState(int i) {
            
//            cout << "+++[DEBUG] _winnersIndices" << endl;
//            for (const auto & myTmpKey : _winnersIndices)
//            {
//                for (const auto & myTmpKey2 : myTmpKey.first)
//                    cout << myTmpKey2 << "-";
//                cout << " -> " << myTmpKey.second << endl;
//            }

            map<vector<int>, int>::const_iterator wIt = _winnersIndices.begin();
            for (; wIt != _winnersIndices.end(); ++wIt) {
                if (wIt->second == i) {
                    return wIt->first;
                }
            }
            assert(false);
        }

        void getClassifiersOutput( vector<int>& classifiersOutput );
		void getCurrentExmapleResult( vector<double>& result );
		
		void setCurrentDataToTrain() { _data->setCurrentDataToTrain(); }
		void setCurrentDataToTest() { _data->setCurrentDataToTest(); }		
        bool setCurrentDataToTest2() { return _data->setCurrentDataToTest2(); }		
		double getAdaboostPerfOnCurrentDataset(){ return _data->getAdaboostPerfOnCurrentDataset(); }
		
		void outPutStatistic(int ep, double acc, double curracc, double uc, double sumrew );
        void outPutStatistic( BinaryResultStruct& bres );
        double getClassificationCost() ;
        
        const ExampleResults* getCurrentExampleResults() { return _exampleResult; }
        
		// constructor
		AdaBoostMDPClassifierContinous(const nor_utils::Args& args, int verbose, DataReader* datareader, int classNum, int discState = 1);
		// destructor
		virtual	~AdaBoostMDPClassifierContinous() 
		{
			_outputStream.close();
		}
		
		///returns the reward for the transition, implements the CRewardFunction interface
		virtual	double	getReward( CStateCollection	*oldState , CAction *action , CStateCollection *newState);
		
		///fetches the internal state and stores it in the state object
		virtual void getState(CState *state); ///resets the model 
		virtual	void doResetModel();		
		
		// get the discretized state space
        virtual CStateModifier* getStateSpace(int);
		virtual CStateModifier* getStateSpaceRBF(unsigned int partitionNumber);
		virtual CStateModifier* getStateSpaceTileCoding(unsigned int partitionNumber);
		virtual CStateModifier* getStateSpaceExp( int divNum, int e );
        virtual CStateModifier* getStateSpaceForGSBNFQFunction(int numOfFeatures);

        virtual void outHeader();
        
        int getPositiveLabelIndex();
        
		// classify correctly
		bool classifyCorrectly();
		bool hasithLabelCurrentElement( int i );
        
        double getIterationError(int it) { return _data->getIterationError(it); }
    
        vector<Label>& getLabels(int i) { return _data->getLabels(i); };
        
        vector<int>& getBagCardinals() { return _data->getBagCardinals(); }
        
        bool isMILsetup() {return _data->isMILsetup(); }
	};
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////					
//	class AdaBoostMDPClassifierContinousEvaluator : public CRewardPerEpisodeCalculator
//	{
//	public:
//		AdaBoostMDPClassifierContinousEvaluator(CAgent *agent, CRewardFunction *rewardFunction) : CRewardPerEpisodeCalculator( agent, rewardFunction, 1000, 2000 )
//		{
//		}
//		
//		double classficationAccruacy( double& acc, double& usedClassifierAvg, const char* logFileName = NULL)
//		{
//			double value = 0;
//			
//			agent->addSemiMDPListener(this);
//			
//			CAgentController *tempController = NULL;
//			if (controller)
//			{
//				tempController = detController->getController();
//				detController->setController(controller);	
//			}
//			
//			AdaBoostMDPClassifierContinous* classifier = dynamic_cast<AdaBoostMDPClassifierContinous*>(semiMDPRewardFunction);
//			const int numTestExamples = classifier->getNumExamples();
//			const int numClasses = classifier->getNumClasses();
//			int  correct = 0, notcorrect = 0;
//			usedClassifierAvg=0;
//			
//			//ofstream output( logFileName );
//			
//			//cout << "Output classfication reult: " << logFileName << endl;
//			ofstream output;
//			vector<double> currentVotes(0);
//			vector<bool> currentHistory(0);
//			
//			if ( logFileName )
//			{
//				output.open( logFileName );			
//				cout << "Output classfication reult: " << logFileName << endl;
//			}
//			
//			for (int i = 0; i < numTestExamples; i ++)
//			{
//				
//				
//				//cout << i << endl;
//				agent->startNewEpisode();				
//				//cout << "Length of history: " << classifier->getLengthOfHistory() << endl;
//				classifier->setCurrentRandomIsntace(i);
//				agent->doControllerEpisode(1, classifier->getIterNum()*2 );
//				//cout << "Length of history: " << classifier->getLengthOfHistory() << endl;
//				
//				//cout << "Intance: " << i << '\t' << "Num of classifier: " << classifier->getUsedClassifierNumber() << endl;
//				bool clRes = classifier->classifyCorrectly();				
//				if (clRes ) correct++;
//				else notcorrect++;
//				
//				usedClassifierAvg += classifier->getUsedClassifierNumber();
//				value += this->getEpisodeValue();
//				
//				//if ((i>10)&&((i%100)==0))
//				//	cout << i << " " << flush;
//				if ( logFileName ) {
//					output << (clRes ? "1" : "0");
//					output << " ";
//					
//					//output << (isNeg ? "1" : "2");
//					output << " ";
//					classifier->getCurrentExmapleResult( currentVotes );
//					classifier->getHistory( currentHistory );
//					for( int l=0; l<numClasses; ++l ) output << currentVotes[l] << " ";
//					//for( int i=0; i<currentHistory.size(); ++i) output << currentHistory[i] << " ";
//					for( int i=0; i<currentHistory.size(); ++i) 
//					{ 
//						if ( currentHistory[i] )
//							output << i+1 << " ";
//					}
//					
//					output << endl << flush;
//				}
//				
//			}
//			
//			cout << endl;
//			
//			value /= (double)numTestExamples ;
//			usedClassifierAvg /= (double)numTestExamples ;
//			acc = ((double)correct/(double)numTestExamples)*100.0;
//			
//			//output.close();
//			if (logFileName) output.close();
//			
//			agent->removeSemiMDPListener(this);
//			
//			if (tempController)
//			{
//				detController->setController(tempController);
//			}
//			
//			return value;		
//		}
//	};
	////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////		
	//class AdaBoostMDPClassifierContinousWitWHInd : public AdaBoostMDPClassifierContinous
	//{
	//	AdaBoostMDPClassifierContinousWitWHInd(const nor_utils::Args& args, int verbose, DataReader* datareader, int classNum );
	//	virtual ~AdaBoostMDPClassifierContinousWitWHInd() {}
	//};
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	class  AdaBoostMDPClassifierSimpleDiscreteSpace : public CAbstractStateDiscretizer
	{
	protected:
		unsigned int _stateNum;
		
	public:
		AdaBoostMDPClassifierSimpleDiscreteSpace(unsigned int stateNum) : CAbstractStateDiscretizer(stateNum), _stateNum( stateNum ) {}
		virtual ~AdaBoostMDPClassifierSimpleDiscreteSpace() {};
		
		virtual unsigned int getDiscreteStateNumber(CStateCollection *state)
		{
			int stateIndex = state->getState()->getDiscreteState(0);
			return stateIndex;			
		}
	};
	
    ////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////
    
	template <typename T>
	class AdaBoostMDPBinaryDiscreteEvaluator : public CRewardPerEpisodeCalculator
	{
	public:
		AdaBoostMDPBinaryDiscreteEvaluator(CAgent *agent, CRewardFunction *rewardFunction ) : CRewardPerEpisodeCalculator( agent, rewardFunction, 1000, 2000 )
		{
		}
		
		void classficationPerformance( BinaryResultStruct& binRes, const string &logFileName, bool detailed = false )
		{
			double value = 0.0;
//            double negNumEval = 0.0;
            
            double classificationCost = 0.;
            
			agent->addSemiMDPListener(this);
			
			CAgentController *tempController = NULL;
			if (controller)
			{
				tempController = detController->getController();
				detController->setController(controller);
			}
			
			T* classifier = dynamic_cast<T*>(semiMDPRewardFunction);

            const int numClasses = classifier->getNumClasses();
			const int numTestExamples = classifier->getNumExamples();
			int  correct = 0, notcorrect = 0;
			int usedClassifierAvg=0;
//			int correctP=0;
//			int posNum=0;
//			int correctN=0;
//			int negNum=0;
			ofstream output;
            ofstream detailedOutput;
			vector<AlphaReal> currentVotes(0);
			vector<bool> currentHistory(0);
			
			if ( !logFileName.empty() )
			{
				output.open( logFileName.c_str() );
                
                if (! output.good()) {
                    cout << "Error! Could not open the log file: " << logFileName << endl;
                    exit(1);
                }

				cout << "Output classfication result: " << logFileName << endl;
                
                if ( detailed ) {
                    string logFileNameDetailed = logFileName + ".detailed";
                    detailedOutput.open(logFileNameDetailed.c_str());
                }
			}
            
            bool milSetup = classifier->isMILsetup();
            
            vector<vector<AlphaReal> > scores;
            if (milSetup) {
                scores.resize(numTestExamples);
            }
			
            
			for (int i = 0; i < numTestExamples; i ++)
			{
				
				//cout << i << endl;
				agent->startNewEpisode();
				//cout << "Length of history: " << classifier->getLengthOfHistory() << endl;
				classifier->setCurrentRandomIsntace(i);
				agent->doControllerEpisode(1,  classifier->getIterNum()*2 );
				//cout << "Length of history: " << classifier->getLengthOfHistory() << endl;
				
				//cout << "Intance: " << i << '\t' << "Num of classifier: " << classifier->getUsedClassifierNumber() << endl;
				bool clRes = classifier->classifyCorrectly();
				if (clRes ) correct++;
				else notcorrect++;
                
//                bool isNeg;
//                if (numClasses <= 2) {
//                    isNeg = !classifier->hasithLabelCurrentElement(classifier->getPositiveLabelIndex());
//                    if (isNeg) // neg
//                    {
//                        negNum++;
//                        if (clRes ) correctN++;
//                    } else {
//                        posNum++;
//                        if (clRes ) correctP++;
//                    }
//                }
				
				classificationCost += classifier->getClassificationCost();
                double numEval = classifier->getUsedClassifierNumber();
				usedClassifierAvg += numEval;
				value += this->getEpisodeValue();
				
                //                if (isNeg) {
                //                    negNumEval += numEval;
                //                }
				
                classifier->getCurrentExmapleResult( currentVotes );
				if ( !logFileName.empty() ) {
                    if (clRes)
                        output << "1" ;
                    else
                        output << "0" ;
                    
//					output << (clRes ? "1" : "0");
					output << " ";
//					output << (isNeg ? "2" : "1");
                    
                    vector<int> classes;
                    vector<Label>& labels = classifier->getLabels(i);
                    for (vector<Label>::iterator lIt = labels.begin(); lIt != labels.end(); ++lIt) {
                        if (lIt->y > 0) classes.push_back(lIt->idx);
                    }
                    
                    
					classifier->getHistory( currentHistory );
                    
                    if (numClasses <= 2) {
//                    if (classes.size() == 1) {
//                        assert(classes[0] == classifier->getPositiveLabelIndex());
                        output << classes[0] << " ";
                        output << currentVotes[classifier->getPositiveLabelIndex()] << " ";
                    }
                    else
                    {
                        for( int l = 0; l < numClasses; ++l )
                            output << currentVotes[l] << " ";
                    }
                    
					for( int wl = 0; wl < currentHistory.size(); ++wl)
					{
						if ( currentHistory[wl] )
							output << wl+1 << " ";
					}
                    
					output << endl << flush;
                    
                    if (detailed) {
                        vector<int> classifiersOutput;
                        classifier->getClassifiersOutput(classifiersOutput);
                        
                        if (clRes)
                            detailedOutput << "1" ;
                        else
                            detailedOutput << "0" ;
                        
                        detailedOutput << " " << classes[0] << " ";
                        
                        for (int i = 0; i < classifiersOutput.size(); ++i) {
                            detailedOutput << classifiersOutput[i] << " ";
                        }
                        detailedOutput << endl << flush;
                    }
                    

				}
				
				//if ((i>10)&&((i%100)==0))
				//	cout << i << " " << flush;
                
                if (milSetup) {
                    scores[i] = currentVotes;
//                    copy(currentVotes.begin(), currentVotes.end(), scores[i].begin());
                }
				
			}
						
            if (milSetup) {
                
                binRes.milError = computeMILError(scores, classifier->getBagCardinals());
            }
            
			binRes.avgReward = value/(double)numTestExamples ;
			binRes.usedClassifierAvg = (double)usedClassifierAvg/(double)numTestExamples ;
            //			binRes.negNumEval = (double)negNumEval/(double)negNum;
            
            binRes.classificationCost = classificationCost/(double)numTestExamples;
            
            binRes.err = ((double)notcorrect/(double)numTestExamples);//*100.0;
			
//			binRes.TP = (double)correctP/(double)posNum;
//			binRes.TN = (double)correctN/(double)negNum;
			
            binRes.itError = classifier->getIterationError((int)binRes.usedClassifierAvg);
            
			//cout << posNum << " " << negNum << endl << flush;
			if (!logFileName.empty()) output.close();
			
			agent->removeSemiMDPListener(this);
			
			if (tempController)
			{
				detController->setController(tempController);
			}
			
		}
        
        // -----------------------------------------------------------------------------------
        
        double computeMILError(vector<vector<AlphaReal> >& g, vector<int>& bagCardinals)
        {
            T* classifier = dynamic_cast<T*>(semiMDPRewardFunction);
            
			const int numExamples = classifier->getNumExamples();
            
            int eventNumber = 0;
            const long numEvents = bagCardinals.size();
            
            vector<int> eventsLabels(numEvents);
            vector<int> eventsOutputLabels(numEvents);
            
            int candidateCounter = 0;
            int numErrors = 0;

//            assert(g.size() == numExamples);
//            cout << "+++[DEBUG]  "  << endl;;
//            for (int i = 0; i < 10; ++i) {
//                for (int j = 0; j < g[i].size(); ++j) {
//                    cout << g[i][j] << " ";
//                }
//                cout << endl;
//            }
//            cout << "+++[DEBUG]  "  << endl;
            
            vector<Label>::const_iterator lIt;
            int i = 0;
            while (i < numExamples)
            {
                int numCandidates = bagCardinals[eventNumber];
                candidateCounter += numCandidates;
                
                const vector<Label>& evtLabels = classifier->getLabels(i);
                for (lIt = evtLabels.begin(); lIt !=  evtLabels.end(); ++lIt) {
                    if (lIt->y > 0) {
                        eventsLabels[eventNumber] = lIt->idx;
                    }
                }
                
                AlphaReal maxClass = -numeric_limits<AlphaReal>::max();
                
                for (int j = 0; j < numCandidates; ++j, ++i) {
                    
                    const vector<Label>& candidatelabels = classifier->getLabels(i);
                    
                    for (lIt = candidatelabels.begin(); lIt !=  candidatelabels.end(); ++lIt) {
                        if (g[i][lIt->idx] > maxClass) {
                            maxClass = g[i][lIt->idx];
                            eventsOutputLabels[eventNumber] = lIt->idx;
                        }
                        
                        if (lIt->y > 0) {
                            assert(eventsLabels[eventNumber] == lIt->idx); // just for precaution
                        }
                    }
                }
                
                if (eventsOutputLabels[eventNumber] != eventsLabels[eventNumber]) {
                    ++numErrors;
                }
                
                ++eventNumber;
            }
            
            //TODO: check if the label is background. Do we take the min ?
            
            assert(candidateCounter == numExamples);
            
//            vector<double> truePositiveRate(numClasses);
//            vector<double> falsePositiveRate(numClasses);
//            
//            vector<int> truePositives(numClasses);
//            vector<int> falsePositives(numClasses);
//            
//            vector<int> numEventsPerClass(numClasses);
//            
//            for (int i = 0; i < numEvents; ++i) {
//                numEventsPerClass[eventsLabels[i]]++;
//                if (eventsLabels[i] == eventsOutputLabels[i])
//                    truePositives[eventsLabels[i]]++;
//                else
//                    falsePositives[eventsOutputLabels[i]]++;
//            }
//            
//            for (int l = 0; l < numClasses; ++l) {
//                truePositiveRate[l] = (double)truePositives[l]/numEventsPerClass[l];
//                falsePositiveRate[l] = (double)falsePositives[l]/(numEvents - numEventsPerClass[l]);
//            }
            
            //        for (int l = 0; l < numClasses; ++l) {
            //            if (l != 0) outStream << OUTPUT_SEPARATOR;
            //            outStream <<  truePositiveRate[l];
            //        }
            //
            //        outStream << OUTPUT_SEPARATOR;
            //        
            //        for (int l = 0; l < numClasses; ++l) {
            //            outStream << OUTPUT_SEPARATOR;
            //            outStream <<  falsePositiveRate[l];
            //        }
            
            return (double)numErrors / numEvents;
        }

	};
    
	

	////////////////////////////////////////////////////////////////////////////////////////////////		
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	
	
} // end of namespace MultiBoost

#endif // __ADABOOST_MDP_CLASSIFIER_ADV_H

