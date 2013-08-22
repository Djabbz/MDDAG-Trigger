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
		
		// loads them
		us.loadHypotheses(shypFileName, _weakHypotheses, _pTrainData);			
		if (_numIterations < _weakHypotheses.size())
			_weakHypotheses.resize(_numIterations);
		
        else _numIterations = _weakHypotheses.size();
        
		if (_verbose > 0)
			cout << "(" << _weakHypotheses.size() << " weak hypotheses kept)" << endl << endl;
		
		assert( _weakHypotheses.size() >= _numIterations );
		
//        cout << "[+++] SUPER SHUFFLE... ACTION! [+++]" << endl;
//        // random shuffle on the shyp
//        random_shuffle ( _weakHypotheses.begin(), _weakHypotheses.end() );
//        random_shuffle ( _weakHypotheses.begin(), _weakHypotheses.end() );

//        cout << "[+++] SUPER REVERSE... ACTION! [+++]" << endl;
//        vector<BaseLearner*> inveresedWhyp(_weakHypotheses.size());
//        copy(_weakHypotheses.rbegin(), _weakHypotheses.rend(), inveresedWhyp.begin());
//        copy(inveresedWhyp.begin(), inveresedWhyp.end(), _weakHypotheses.begin());
        
		// calculate the sum of alphas
		vector<BaseLearner*>::iterator it;
		_sumAlphas=0.0;
		for( it = _weakHypotheses.begin(); it != _weakHypotheses.end(); ++it )
		{
			BaseLearner* currBLearner = *it;
			_sumAlphas += currBLearner->getAlpha();			
		}
	
		_isDataStorageMatrix = false;
		 
		if (_isDataStorageMatrix)
		{			
			setCurrentDataToTrain();
			calculateHypothesesMatrix();
			setCurrentDataToTest();
			calculateHypothesesMatrix();
			
			_vs.resize(_numIterations);
			_alphas.resize(_numIterations);
			for(int wHypInd = 0; wHypInd < _numIterations; ++wHypInd )
			{								
				AbstainableLearner* currWeakHyp = dynamic_cast<AbstainableLearner*>(_weakHypotheses[wHypInd]);
				_alphas[wHypInd] =  currWeakHyp->getAlpha();
				_vs[wHypInd] = currWeakHyp->_v;
				for(int l=0; l<_vs[wHypInd].size(); ++l)
				{
					_vs[wHypInd][l] *= currWeakHyp->getAlpha();
				}
			}
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
            
            if (! testFileName2.empty())
            {
                _bagCardinals[_pTestData2] = vector<int>();
                readRawData(testFileName2 + ".events", _bagCardinals[_pTestData2]);
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
		cout << "Calculate weak hyp matrix ...";
		const int numExamples = _pCurrentData->getNumExamples();

		vVecChar& tmpOutput = _weakHypothesesMatrices[_pCurrentData];
		tmpOutput.resize(numExamples);
		
		for( int i=0; i<numExamples; ++i )
		{
			tmpOutput[i].resize(_numIterations);
		}		
		
		for(int wHypInd = 0; wHypInd < _numIterations; ++wHypInd )
		{
			AbstainableLearner* currWeakHyp = dynamic_cast<AbstainableLearner*>(_weakHypotheses[wHypInd]);
			vector<AlphaReal> tmpv = currWeakHyp->_v;
			
			for( int i=0; i<numExamples; ++i )
			{				
				AlphaReal cl = currWeakHyp->classify( _pCurrentData, i, 0 );								
				cl *= tmpv[0];
				tmpOutput[i][wHypInd] = (cl<0.0) ? -1.0 : +1.0;
//				if (cl>0.0) tmpBitSet[i].set(wHypInd,1); // +1
//				else tmpBitSet[i].set(wHypInd,0); // -1
			}
		}
								
		cout << "Done" << endl;
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

    void DataReader::reorderWeakHypotheses(InputData* pData){
        
        vector<BaseLearner*> reorderedWhyp;
        vector<BaseLearner*>::iterator whyIt;
        
        const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();

        vector<vector<AlphaReal> > posteriors(numExamples);
        for (int i = 0; i < numExamples; ++i)
            posteriors[i].resize(numClasses);
        
        for (int i = 0; i < numExamples; ++i)
        {
            for (whyIt = _weakHypotheses.begin(); whyIt != _weakHypotheses.end(); ++whyIt) {
                BaseLearner* currWeakHyp = *whyIt;
                AlphaReal alpha = currWeakHyp->getAlpha();
            
                for (int l = 0; l < numClasses; ++l)
                    posteriors[i][l] += alpha * currWeakHyp->classify(pData, i, l);
            }

        }
        
        cout << "[+] Reordering the base classifiers... " << flush;
        
        for (whyIt = _weakHypotheses.begin(); whyIt != _weakHypotheses.end(); ++whyIt) {
            AlphaReal span = computeSeparationSpan(pData, posteriors);
        }
        
        
        cout << "Done!" << endl;
        
    }
    
    
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
			//vBitSet& cBitSet = _weakHypothesesMatrices[_pCurrentData];
			//for (int l = 0; l < numClasses; ++l)
				//currVotesVector[l] += alpha * _vs[wHypInd][l] * (_pCurrentBitset->at(instance)[wHypInd]? +1.0 : -1.0);
			alpha = _alphas[wHypInd];
			for (int l = 0; l < numClasses; ++l) {
				currVotesVector[l] += _vs[wHypInd][l] * (*_pCurrentMatrix)[instance][wHypInd];
                //TODO: To be checked
                ternaryPhis[l] = (*_pCurrentMatrix)[instance][wHypInd];
            }
            
			
		} else
		{
			BaseLearner* currWeakHyp = _weakHypotheses[wHypInd];
			alpha = currWeakHyp->getAlpha();
            
//            vote = currWeakHyp->classify(_pCurrentData, instance, 0);

			for (int l = 0; l < numClasses; ++l) {
                int vote = currWeakHyp->classify(_pCurrentData, instance, l);
				currVotesVector[l] += alpha * vote;
                ternaryPhis[l] = (vote > 0) ? 1 : ((vote < 0) ? -1 : 0) ;
            }
		}
		
		return ternaryPhis;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
    
    AlphaReal DataReader::getWhypClassification( const int wHypInd, const int instance )
	{
		const int numClasses = _pCurrentData->getNumClasses();
		
        BaseLearner* currWeakHyp = _weakHypotheses[wHypInd];
        AlphaReal alpha = currWeakHyp->getAlpha();
        int vote = currWeakHyp->classify(_pCurrentData, instance, 0);
        
        vector<AlphaReal> scoreVector(numClasses);
        for (int l = 0; l < numClasses; ++l)
            scoreVector[l] = alpha * currWeakHyp->classify(_pCurrentData, instance, l);
		
		return alpha * vote;
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

            BaseLearner* currWeakHyp = _weakHypotheses[j];
            AlphaReal alpha = currWeakHyp->getAlpha();

            for( int i = 0; i < numExamples; ++i )
            {
                ExampleResults*& tmpResult = examplesResults[i];
                vector<AlphaReal>& currVotesVector = tmpResult->getVotesVector();
            
                // for every class
                for (int l = 0; l < numClasses; ++l)
                    currVotesVector[l] += alpha * currWeakHyp->classify(_pCurrentData, i, l);
                
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