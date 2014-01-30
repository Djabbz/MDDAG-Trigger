/*
 *  AdaBoostMDPClassifierAdv.cpp
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/10/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaBoostMDPClassifierAdv.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "WeakLearners/FeaturewiseLearner.h"
#include "WeakLearners/AbstainableLearner.h"

#include <math.h> // for exp
#include <algorithm> // for random_shuffle

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	// -----------------------------------------------------------------------

	// -----------------------------------------------------------------------
	DataReader::DataReader(const nor_utils::Args& args, int verbose) : _verbose(verbose), _args(args), _pTestData2(NULL)
	{				
        int optionsCursor = 0;
		string mdpTrainFileName = _args.getValue<string>("traintestmdp", optionsCursor);	
        ++optionsCursor;
		string testFileName = _args.getValue<string>("traintestmdp", optionsCursor);				
        ++optionsCursor;
        
		_numIterations = _args.getValue<int>("traintestmdp", optionsCursor);				
        ++optionsCursor;		
        string shypFileName = _args.getValue<string>("traintestmdp", optionsCursor);
        ++optionsCursor;
		string tmpFname = _args.getValue<string>("traintestmdp", optionsCursor);		
        ++optionsCursor;
        string testFileName2;
        if (_args.getNumValues("traintestmdp") > 5) {
            testFileName2 = _args.getValue<string>("traintestmdp", optionsCursor);
        }
        
        
		if (_verbose > 0)
			cout << "Loading arff data for MDP learning..." << flush;
		
		// load the arff
		loadInputData(mdpTrainFileName, testFileName, testFileName2, shypFileName);
		
		if (_verbose > 0)
			cout << "Done." << endl << flush;
		
		
		if (_verbose > 0)
			cout << "Loading strong hypothesis..." << flush;
		
		// The class that loads the weak hypotheses
		UnSerialization us;
		
        vector<BaseLearner*>	weakHypotheses;
        
		// loads them
		us.loadHypotheses(shypFileName, weakHypotheses, _pTrainData);
		if (_numIterations < weakHypotheses.size())
			weakHypotheses.resize(_numIterations);
		
        else _numIterations = (int)weakHypotheses.size();
        
        _totalNumIterations = _numIterations;
        
		if (_verbose > 0)
			cout << "(" << weakHypotheses.size() << " weak hypotheses kept)" << endl << endl;
		
		assert( weakHypotheses.size() >= _numIterations );
		
//        cout << "[+++] SUPER SHUFFLE... ACTION! [+++]" << endl;
//        // random shuffle on the shyp
//        random_shuffle ( _weakHypotheses.begin(), _weakHypotheses.end() );
//        random_shuffle ( _weakHypotheses.begin(), _weakHypotheses.end() );

//        cout << "[+++] SUPER REVERSE... ACTION! [+++]" << endl;
//        vector<BaseLearner*> inveresedWhyp(_weakHypotheses.size());
//        copy(_weakHypotheses.rbegin(), _weakHypotheses.rend(), inveresedWhyp.begin());
//        copy(inveresedWhyp.begin(), inveresedWhyp.end(), _weakHypotheses.begin());

//        cout << "[+++] SUPER REORDER ! BY FEATURE COST... ACTION! [+++]" << endl;
        
        
//        cheap_vars.push_back("D0_VTX_FD");  D0_VTX_FD PiS_IP PiS_IPC2 D0C_1_IP D0C_1_IPC D0C_2_IP D0C_2_IPC
//        cheap_vars.push_back("PiS_IP");
//        cheap_vars.push_back("PiS_IPC2");
//        cheap_vars.push_back("D0C_1_IP");
//        cheap_vars.push_back("D0C_1_IPC");
//        cheap_vars.push_back("D0C_2_IP");
//        cheap_vars.push_back("D0C_2_IPC");

		// calculate the sum of alphas
		vector<BaseLearner*>::iterator it;
		_sumAlphas=0.0;
        
        cout << "[+] Summing the alpha... " << flush;
		for( it = weakHypotheses.begin(); it != weakHypotheses.end(); ++it )
		{
			BaseLearner* currBLearner = *it;
			_sumAlphas += currBLearner->getAlpha();			
		}
        cout << "done!" << endl;
        
        _groupedFeatures = false;
        if (args.hasArgument("groupedfeatures"))
        {
            _groupedFeatures = true;
        }
        
        if (_groupedFeatures) {

            cout << "[+] Grouped features \n\t" << flush;

            map<set<int>, vector<BaseLearner*> > featureWhypMap;
            map<set<int>, vector<BaseLearner*> >::iterator fwIt;

            ofstream featureFile;
            featureFile.open("grouped_features.dta");
            
            const NameMap& attributeNamemap = _pTrainData->getAttributeNameMap();

            if (args.hasArgument("budgeted") && args.getValue<string>("budgeted", 0).compare("LHCb") == 0)
            {
                vector<string> cheap_vars;
                if (cheap_vars.size() == 0)
                {
                    cheap_vars.push_back("D0_VTX_FD");
                    cheap_vars.push_back("PiS_IP");
                    cheap_vars.push_back("D0C_1_IP");
                    cheap_vars.push_back("D0C_2_IP");
                    cheap_vars.push_back("D0Tau");
                }
                
                set<int> cheapVarIndices;
                vector<string>::iterator varIt;
                for (varIt = cheap_vars.begin(); varIt != cheap_vars.end() ; ++varIt)
                {
                    int var_index = attributeNamemap.getIdxFromName(*varIt);
                    cheapVarIndices.insert(var_index);
                }
                
                for( it = weakHypotheses.begin(); it != weakHypotheses.end(); ++it )
                {
                    set<int> featuresUsed = (*it)->getUsedColumns();
                    
                    bool nonCheapFeaturesUsed = false;
                    for (set<int>::iterator idxIt = featuresUsed.begin(); idxIt != featuresUsed.end(); ++idxIt)
                    {
                        if (cheapVarIndices.find(*idxIt) == cheapVarIndices.end())
                        {
                            nonCheapFeaturesUsed = true;
                            break;
                        }
                    }
                    
                    if (nonCheapFeaturesUsed)
                        featureWhypMap[featuresUsed].push_back(*it);
                    else
                        featureWhypMap[cheapVarIndices].push_back(*it);
                }
                
                // filling the weakhyp vector<vector>
                
                _weakHypotheses.push_back(featureWhypMap[cheapVarIndices]);
                for (set<int>::iterator idxIt = cheapVarIndices.begin(); idxIt != cheapVarIndices.end(); ++idxIt) {
                    cout << *idxIt << ", ";
                }
                cout << "\t -> " << featureWhypMap[cheapVarIndices].size() << "\n\t";

                featureWhypMap.erase(cheapVarIndices);
                
                // manual reordering
                int feat_indices[] = {4, 8, 12, 0, 1, 7, 11, 15, 6, 10, 14};
                for (int f = 0; f < 11; ++f) {
                    set<int> f_set;
                    f_set.insert(feat_indices[f]);
                    if (featureWhypMap[f_set].size() != 0)
                        _weakHypotheses.push_back(featureWhypMap[f_set]);
                    cout << feat_indices[f] << "\t -> " << featureWhypMap[f_set].size() << "\n\t";
                    featureWhypMap.erase(f_set);
                    f_set.erase(feat_indices[f]);
                }
                
                for (fwIt = featureWhypMap.begin(); fwIt != featureWhypMap.end(); ++fwIt) {
                    for (set<int>::iterator idxIt = fwIt->first.begin(); idxIt != fwIt->first.end(); ++idxIt) {
                        cout << *idxIt << ", ";
                    }
                    cout << "\t -> " << fwIt->second.size() << "\n\t";
                    _weakHypotheses.push_back(fwIt->second);
                }
                
                cout << endl;
            }
            else
            {
                for( it = weakHypotheses.begin(); it != weakHypotheses.end(); ++it )
                {
                    set<int> featuresUsed = (*it)->getUsedColumns();
                    featureWhypMap[featuresUsed].push_back(*it);
                }
                
                for (fwIt = featureWhypMap.begin(); fwIt != featureWhypMap.end(); ++fwIt) {
                    for (set<int>::iterator idxIt = fwIt->first.begin(); idxIt != fwIt->first.end(); ++idxIt) {
                        cout << *idxIt << ", ";
                    }
                    cout << "\t -> " << fwIt->second.size() << "\n\t";
                    _weakHypotheses.push_back(fwIt->second);
                }
            }
            
            _numIterations = (int)_weakHypotheses.size();
            
            // logging
            cout << "[+] Logging the grouped features... " << flush;
            for (int i = 0; i < _numIterations; ++i) {
                for (int j = 0; j < _weakHypotheses[i].size(); ++j) {
                    
                    featureFile << _weakHypotheses[i][j]->index;
                    
                    set<int> featuresUsed = _weakHypotheses[i][j]->getUsedColumns();
                    for (set<int>::iterator idxIt = featuresUsed.begin(); idxIt != featuresUsed.end(); ++idxIt)
                    {
                        featureFile << "," << attributeNamemap.getNameFromIdx(*idxIt);
                    }
                    
                    featureFile << " ";
                }
                featureFile << "\n";
            }
            featureFile.close();
            cout << "Done." << endl;
        }
        else
        {
            _weakHypotheses.resize(weakHypotheses.size());
            for(int i = 0; i < weakHypotheses.size(); ++i)
            {
                _weakHypotheses[i].push_back(weakHypotheses[i]);
            }
        }

		_isDataStorageMatrix = false;
        if (args.hasArgument("cpuopt"))
        {
            _isDataStorageMatrix = true;
        }
		 
		if (_isDataStorageMatrix)
		{			
			setCurrentDataToTrain();
			calculateHypothesesMatrix();
			setCurrentDataToTest();
			calculateHypothesesMatrix();
			
//			_vs.resize(_numIterations);
//			_alphas.resize(_numIterations);
//			for(int wHypInd = 0; wHypInd < _numIterations; ++wHypInd )
//			{								
//				AbstainableLearner* currWeakHyp = dynamic_cast<AbstainableLearner*>(weakHypotheses[wHypInd]);
//				_alphas[wHypInd] =  currWeakHyp->getAlpha();
//				_vs[wHypInd] = currWeakHyp->_v;
//				for(int l=0; l<_vs[wHypInd].size(); ++l)
//				{
//					_vs[wHypInd][l] *= currWeakHyp->getAlpha();
//				}
//			}
		}
        
        // for budgetted classification
        _mil = false;
        
        if (args.hasArgument("mil"))
        {
            _mil = true;
            
            _bagCardinals[_pTrainData] = vector<int>();
            readRawData(mdpTrainFileName + ".events", _bagCardinals[_pTrainData]);            
            
            _bagCardinals[_pTestData] = vector<int>();
            readRawData(testFileName + ".events", _bagCardinals[_pTestData]);
            
            int counter;
            
            counter = 0;
            _bagOffsets[_pTrainData] = vector<int>();
            _bagOffsets[_pTrainData].resize(_bagCardinals[_pTrainData].size());
            for (int i = 0; i < _bagCardinals[_pTrainData].size(); ++i) {
                _bagOffsets[_pTrainData][i] = counter;
                counter += _bagCardinals[_pTrainData][i];
            }

            counter = 0;
            _bagOffsets[_pTestData] = vector<int>();
            _bagOffsets[_pTestData].resize(_bagCardinals[_pTestData].size());
            for (int i = 0; i < _bagCardinals[_pTestData].size(); ++i) {
                _bagOffsets[_pTestData][i] = counter;
                counter += _bagCardinals[_pTestData][i];
            }

            if (! testFileName2.empty())
            {
                _bagCardinals[_pTestData2] = vector<int>();
                readRawData(testFileName2 + ".events", _bagCardinals[_pTestData2]);
                
                _bagOffsets[_pTestData2] = vector<int>();
                counter = 0;
                for (int i = 0; i < _bagCardinals[_pTestData2].size(); ++i) {
                    _bagOffsets[_pTestData2][i] = counter;
                    counter += _bagCardinals[_pTestData2][i];
                }
            }
        }
	}
    
	// -----------------------------------------------------------------------
    
    void DataReader::readRawData(string rawDataFileName, vector<int>& candidateNumber)
    {
        candidateNumber.clear();
        
        ifstream ifs(rawDataFileName.c_str());
        if (! ifs.good() ) {
            cout << "Error: Could not open event file: " << rawDataFileName << endl;
            exit(1);
        }
        
        while (ifs.good()) {
            int i ;
            ifs >> i;
            candidateNumber.push_back(i);
        }
    }

	// -----------------------------------------------------------------------

	void DataReader::calculateHypothesesMatrix()
	{		
		cout << "Calculate weak hyp matrix " << flush;
		const int numExamples = _pCurrentData->getNumExamples();
        const int numClasses = _pCurrentData->getNumClasses();
        
		hypermat& allOutputs = _weakHypothesesMatrices[_pCurrentData];
		allOutputs.resize(numExamples);
		
		for(int i = 0; i < numExamples; ++i)
		{
			allOutputs[i].resize(_numIterations);
            for (int j = 0; j < _numIterations; ++j) {
                allOutputs[i][j].resize(numClasses, 0.);
            }
		}		
		
        const int step = _totalNumIterations < 10 ? 1 : _totalNumIterations / 10;
    
        cout << ": 0%." << flush;
        int t = 0;
		for(int wHypInd = 0; wHypInd < _numIterations; ++wHypInd )
		{
            if ((t + 1) % 1000 == 0)
                cout << "." << flush;

            if ((t + 1) % step == 0)
            {
                float progress = static_cast<float>(t) / static_cast<float>(_totalNumIterations) * 100.0;
                cout << "." << setprecision(2) << progress << "%." << flush;
            }


            vector<BaseLearner*>::iterator whypIt;
            for (whypIt = _weakHypotheses[wHypInd].begin(); whypIt != _weakHypotheses[wHypInd].end(); ++whypIt) {
//                AbstainableLearner* currWeakHyp = dynamic_cast<AbstainableLearner*>(*whypIt);
                BaseLearner* currWeakHyp = *whypIt;
                AlphaReal alpha = currWeakHyp->getAlpha();
                
                for(int i = 0; i < numExamples; ++i)
                {
                    for (int l = 0; l < numClasses; ++l)
                    {
                        allOutputs[i][wHypInd][l] += alpha * currWeakHyp->classify(_pCurrentData, i, l);
                    }
                }
                ++t;
            }
        }
								
		cout << "Done." << endl;
	}
	
    // -----------------------------------------------------------------------
    
    AlphaReal DataReader::computeSeparationSpan(InputData* pData, const vector<vector<AlphaReal> > & iPosteriors)
    {
        const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();
        
        vector<int> numExamplesPerClass(numClasses);
        for (int l = 0; l < numClasses; ++l) {
            numExamplesPerClass[l] = pData->getNumExamplesPerClass(l);
        }
        
        AlphaReal minSpan = numeric_limits<AlphaReal>::max();
        
        // pairwise
//        for (int l1 = 0; l1 < numClasses - 1; ++l1) {
//            for (int l2 = l1 + 1; l2 < numClasses; ++l2) {
//                int numPositiveExamples = numExamplesPerClass[l1];
//                int numNegativeExamples = numExamplesPerClass[l2];
//                AlphaReal edgePos = 0., edgeNeg = 0.;
//                for (int i = 0; i < numExamples; ++i) {
//                    AlphaReal posterior = iPosteriors[i][l1]; [l1][l1][l1][l1]
//                    vector<Label>& labels = pData->getLabels(i);
//                    if (labels[l1].y > 0)
//                        edgePos += posterior;
//                    if (labels[l2].y > 0)
//                        edgeNeg += posterior;
//                }
//                AlphaReal span = edgePos / numPositiveExamples - edgeNeg / numNegativeExamples;
//                if (span < minSpan) {
//                    minSpan = span;
//                }
//            }
//        }
        
        // one against all
        for (int l1 = 0; l1 < numClasses - 1; ++l1) {
            int numPositiveExamples = numExamplesPerClass[l1];
            int numNegativeExamples = numExamples - numPositiveExamples;
            AlphaReal edgePos = 0., edgeNeg = 0.;
            for (int i = 0; i < numExamples; ++i) {
                AlphaReal posterior = iPosteriors[i][l1];
                vector<Label>& labels = pData->getLabels(i);
                if (labels[l1].y > 0)
                    edgePos += posterior;
                else
                    edgeNeg += posterior;
            }
            AlphaReal span = edgePos / numPositiveExamples - edgeNeg / numNegativeExamples;
            if (span < minSpan) {
                minSpan = span;
            }
        }

        return minSpan;
    }

    // -----------------------------------------------------------------------

//    void DataReader::reorderWeakHypotheses(InputData* pData){
//        
//        vector<BaseLearner*> reorderedWhyp;
//        vector<BaseLearner*>::iterator whyIt;
//        
//        const int numExamples = pData->getNumExamples();
//		const int numClasses = pData->getNumClasses();
//
//        vector<vector<AlphaReal> > posteriors(numExamples);
//        for (int i = 0; i < numExamples; ++i)
//            posteriors[i].resize(numClasses);
//        
//        for (int i = 0; i < numExamples; ++i)
//        {
//            for (whyIt = _weakHypotheses.begin(); whyIt != _weakHypotheses.end(); ++whyIt) {
//                BaseLearner* currWeakHyp = *whyIt;
//                AlphaReal alpha = currWeakHyp->getAlpha();
//            
//                for (int l = 0; l < numClasses; ++l)
//                    posteriors[i][l] += alpha * currWeakHyp->classify(pData, i, l);
//            }
//
//        }
//        
//        cout << "[+] Reordering the base classifiers... " << flush;
//        
//        for (whyIt = _weakHypotheses.begin(); whyIt != _weakHypotheses.end(); ++whyIt) {
//            AlphaReal span = computeSeparationSpan(pData, posteriors);
//        }
//        
//        
//        cout << "Done!" << endl;
//        
//    }
    
    
    // -----------------------------------------------------------------------

	void DataReader::loadInputData(const string& dataFileName, const string& testDataFileName, const string& testDataFileName2, const string& shypFileName)
	{
		// open file
		ifstream inFile(shypFileName.c_str());
		if (!inFile.is_open())
		{
			cerr << "ERROR: Cannot open strong hypothesis file <" << shypFileName << ">!" << endl;
			exit(1);
		}
		
		// Declares the stream tokenizer
		nor_utils::StreamTokenizer st(inFile, "<>\n\r\t");
		
		// Move until it finds the multiboost tag
		if ( !UnSerialization::seekSimpleTag(st, "multiboost") )
		{
			// no multiboost tag found: this is not the correct file!
			cerr << "ERROR: Not a valid MultiBoost Strong Hypothesis file: " << shypFileName << endl;
			exit(1);
		}
		
		// Move until it finds the algo tag
		string basicLearnerName = UnSerialization::seekAndParseEnclosedValue<string>(st, "algo");
		
		// Check if the weak learner exists
		if ( !BaseLearner::RegisteredLearners().hasLearner(basicLearnerName) )
		{
			cerr << "ERROR: Weak learner <" << basicLearnerName << "> not registered!!" << endl;
			exit(1);
		}
		
		// get the training input data, and load it
		BaseLearner* baseLearner = BaseLearner::RegisteredLearners().getLearner(basicLearnerName);
		baseLearner->initLearningOptions(_args);
		_pTrainData = baseLearner->createInputData();
		
		// set the non-default arguments of the input data
		_pTrainData->initOptions(_args);
		// load the data
		_pTrainData->load(dataFileName, IT_TEST, _verbose);				
		_pCurrentData = _pTrainData;
		
		_pTestData = baseLearner->createInputData();
		
		// set the non-default arguments of the input data
		_pTestData->initOptions(_args);
		// load the data
		_pTestData->load(testDataFileName, IT_TEST, _verbose);				
        
        
        _pTestData2 = NULL;
        if (!testDataFileName2.empty()) {
            _pTestData2 = baseLearner->createInputData();
            
            // set the non-default arguments of the input data
            _pTestData2->initOptions(_args);
            // load the data
            _pTestData2->load(testDataFileName2, IT_TEST, _verbose);				            
        }        
	}				
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	vector<int> DataReader::classifyKthWeakLearner( const int wHypInd, const int instance, ExampleResults* exampleResult )
	{		
		if (_verbose>3) {
			//cout << "Classifiying: " << wHypInd << endl;
		}
		
		if ( wHypInd >= _numIterations ) {
            assert(false);
        }
		
		const int numClasses = _pCurrentData->getNumClasses();				
		
		// a reference for clarity and speed
		vector<AlphaReal>& currVotesVector = exampleResult->getVotesVector();
        
        vector<int> ternaryPhis(numClasses);
        
		AlphaReal alpha;
		
		// for every class
		if (_isDataStorageMatrix)
		{
			for (int l = 0; l < numClasses; ++l) {
				currVotesVector[l] += (*_pCurrentMatrix)[instance][wHypInd][l];
                ternaryPhis[l] = (currVotesVector[l] > 0) ? 1 : ((currVotesVector[l] < 0) ? -1 : 0) ;
            }
		}
        else
		{
            vector<BaseLearner*>::iterator whypIt;
            for (whypIt = _weakHypotheses[wHypInd].begin(); whypIt != _weakHypotheses[wHypInd].end(); ++whypIt) {
                BaseLearner* currWeakHyp = *whypIt;
                alpha = currWeakHyp->getAlpha();
                
                for (int l = 0; l < numClasses; ++l) {
                    int vote = currWeakHyp->classify(_pCurrentData, instance, l);
                    currVotesVector[l] += alpha * vote;
                    
                    ternaryPhis[l] = (currVotesVector[l] > 0) ? 1 : ((currVotesVector[l] < 0) ? -1 : 0) ;
                }
            }
		}
		
		return ternaryPhis;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
    
    vector<AlphaReal> DataReader::getWhypClassification( const int wHypInd, const int instance )
	{
		const int numClasses = _pCurrentData->getNumClasses();
		
        vector<AlphaReal> scoreVector(numClasses);
        
        vector<BaseLearner*>::iterator whypIt;
        for (whypIt = _weakHypotheses[wHypInd].begin(); whypIt != _weakHypotheses[wHypInd].end(); ++whypIt) {
            BaseLearner* currWeakHyp = *whypIt;

            AlphaReal alpha = currWeakHyp->getAlpha();
            
            for (int l = 0; l < numClasses; ++l)
                scoreVector[l] += alpha * currWeakHyp->classify(_pCurrentData, instance, l);
		}
		return scoreVector;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------

	bool DataReader::currentClassifyingResult( const int currentIstance, ExampleResults* exampleResult )
	{
		vector<Label>::const_iterator lIt;
		
		const vector<Label>& labels = _pCurrentData->getLabels(currentIstance);
		
		// the vote of the winning negative class
		AlphaReal maxNegClass = -numeric_limits<AlphaReal>::max();
		// the vote of the winning positive class
		AlphaReal minPosClass = numeric_limits<AlphaReal>::max();
		
		vector<AlphaReal>& currVotesVector = exampleResult->getVotesVector();
		
		for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
		{
			// get the negative winner class
			if ( lIt->y < 0 && currVotesVector[lIt->idx] > maxNegClass )
				maxNegClass = currVotesVector[lIt->idx];
			
			// get the positive winner class
			if ( lIt->y > 0 && currVotesVector[lIt->idx] < minPosClass )
				minPosClass = currVotesVector[lIt->idx];
		}
		
		if ( nor_utils::is_zero( minPosClass - maxNegClass )) return false;
		
		// if the vote for the worst positive label is lower than the
		// vote for the highest negative label -> error
		if (minPosClass < maxNegClass){
			return false;
		} else {
			return true;
		}
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
    
	AlphaReal DataReader::getExponentialLoss( const int currentIstance, ExampleResults* exampleResult )
	{
		AlphaReal exploss = 0.0;
		
		vector<Label>::const_iterator lIt;
		
		const int numClasses = _pCurrentData->getNumClasses();
		const vector<Label>& labels = _pCurrentData->getLabels(currentIstance);
		vector<AlphaReal> yfx(numClasses);
				
		vector<AlphaReal>& currVotesVector = exampleResult->getVotesVector();
		
		//cout << "Instance: " << currentIstance << " ";
		//cout <<  "Size: " << currVotesVector.size() << " Data: ";
		
		for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
		{
			//cout << currVotesVector[lIt->idx] << " ";
			yfx[lIt->idx] = currVotesVector[lIt->idx] * lIt->y;
		}							
		//cout << endl << flush;
		
		if (numClasses==2) // binary classification
		{
			exploss = exp(-yfx[0]);
		} else {			
			for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				exploss += exp(-yfx[lIt->idx]);
			}						
		}
		
		return exploss;
	}	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
    
	AlphaReal DataReader::getMargin( const int currentIstance, ExampleResults* exampleResult )
	{		
		vector<Label>::const_iterator lIt;
		
		const int numClasses = _pCurrentData->getNumClasses();
		const vector<Label>& labels = _pCurrentData->getLabels(currentIstance);
		vector<AlphaReal> yfx(numClasses);
        
		vector<AlphaReal>& currVotesVector = exampleResult->getVotesVector();
		
        AlphaReal mean = 0;
        int num = 0;
		for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
		{
            yfx[lIt->idx] = currVotesVector[lIt->idx];
            mean += currVotesVector[lIt->idx];
            ++num;
		}
        
        // normalize, because of the weird base classifiers who vote for both classes
        mean /= num;
        for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
		{
			yfx[lIt->idx] -= mean;
            yfx[lIt->idx] *= lIt->y;
		}
		
        AlphaReal minMargin = *min_element(yfx.begin(), yfx.end());

		return minMargin;
	}
    
	// -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    
    
    AlphaReal DataReader::getLogisticLoss( const int currentIstance, ExampleResults* exampleResult )
	{
		AlphaReal logitloss = 0.0;
		
		vector<Label>::const_iterator lIt;
		
		const int numClasses = _pCurrentData->getNumClasses();
		const vector<Label>& labels = _pCurrentData->getLabels(currentIstance);
		vector<AlphaReal> yfx(numClasses);
        
		vector<AlphaReal>& currVotesVector = exampleResult->getVotesVector();
				
		for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
		{
			yfx[lIt->idx] = currVotesVector[lIt->idx] * lIt->y;
		}
		
		if (numClasses==2) // binary classification
		{
			logitloss = log(1 + exp(-yfx[0]));
		} else {
			for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				logitloss += log(1 + exp(-yfx[lIt->idx]));
			}
		}
		
		return logitloss;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------

	bool DataReader::hasithLabel( int currentIstance, int classIdx )
	{
		const vector<Label>& labels = _pCurrentData->getLabels(currentIstance);
		return (labels[classIdx].y>0);
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	double DataReader::getAdaboostPerfOnCurrentDataset()
	{
		const int numClasses = _pCurrentData->getNumClasses();
		const int numExamples = _pCurrentData->getNumExamples();
		
		int correct = 0;
		int incorrect = 0;
		
        double err;
        
        vector<double>& iterationWiseError = _iterationWiseError[_pCurrentData];
        iterationWiseError.resize(_weakHypotheses.size());
        
        vector<ExampleResults*> examplesResults(numExamples);
        for (int i = 0; i < numExamples; ++i)
			examplesResults[i] = new ExampleResults(i, numClasses) ;

        for( int j = 0; j < _weakHypotheses.size(); ++j )
        {
            correct = 0;
            incorrect = 0;

            for( int i = 0; i < numExamples; ++i )
            {
                ExampleResults*& tmpResult = examplesResults[i];
                vector<AlphaReal>& currVotesVector = tmpResult->getVotesVector();
                
                if (_isDataStorageMatrix)
                {
                    for (int l = 0; l < numClasses; ++l)
                        currVotesVector[l] += (*_pCurrentMatrix)[i][j][l];
                }
                else
                {
                    vector<BaseLearner*>::iterator whypIt;
                    for (whypIt = _weakHypotheses[j].begin(); whypIt != _weakHypotheses[j].end(); ++whypIt) {
                        BaseLearner* currWeakHyp = *whypIt;
                        AlphaReal alpha = currWeakHyp->getAlpha();
                        
                        // for every class
                        for (int l = 0; l < numClasses; ++l)
                            currVotesVector[l] += alpha * currWeakHyp->classify(_pCurrentData, i, l);
                    }
                }
                
                vector<Label>::const_iterator lIt;
                const vector<Label>& labels = _pCurrentData->getLabels(i);
                
                // the vote of the winning negative class
                AlphaReal maxNegClass = -numeric_limits<AlphaReal>::max();
                // the vote of the winning positive class
                AlphaReal minPosClass = numeric_limits<AlphaReal>::max();
                
                for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
                {
                    // get the negative winner class
                    if ( lIt->y < 0 && currVotesVector[lIt->idx] > maxNegClass )
                        maxNegClass = currVotesVector[lIt->idx];
                    
                    // get the positive winner class
                    if ( lIt->y > 0 && currVotesVector[lIt->idx] < minPosClass )
                        minPosClass = currVotesVector[lIt->idx];
                }
                
                // if the vote for the worst positive label is lower than the
                // vote for the highest negative label -> error
                if (minPosClass <= maxNegClass)
                    incorrect++;
                else {
                    correct++;
                }
            }
            
            err = ((double) incorrect / ((double) numExamples)); // * 100.0;
            iterationWiseError[j] = err;
		}

//        cout << endl;
//        int i = 0;
//        for (const auto & myTmpKey : _iterationWiseError[_pCurrentData]) {
//            cout << myTmpKey << " ";
//            ++i;
//            if (i > 50) {
//                break;
//            }
//        }
//        cout << endl;
        
        for (int i = 0; i < numExamples; ++i)
			delete examplesResults[i] ;
        

//		double acc = ((double) correct / ((double) numExamples)) * 100.0;
		return err;
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	AdaBoostDiscreteState::AdaBoostDiscreteState(unsigned int iterNum, unsigned int classNum) : CAbstractStateDiscretizer((iterNum+1) * (2^classNum))
	{
		this->_iterNum = iterNum;
		this->_classNum = classNum;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	unsigned int AdaBoostDiscreteState::getDiscreteStateNumber(CStateCollection *state) {
		unsigned int discstate=0;		
		int iter = state->getState()->getDiscreteState(0);
		
		if (iter < 0 || (unsigned int)iter > _iterNum )
		{
			discstate = 0;
		}
		else
		{
			int base = 0;
			for( int i = 1; i <= _classNum; ++i )
			{
				int v = state->getState()->getDiscreteState(i);
				base += 2^(i-1) * v;				
			}
			discstate = base * (_iterNum+1) + iter;
		}
		return discstate;
	}
	
	
	
	
	
	
} // end of namespace MultiBoost