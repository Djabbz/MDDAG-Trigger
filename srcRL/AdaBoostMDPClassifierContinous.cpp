/*
 *  AdaBoostMDPClassifierContinous.cpp
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/11/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaBoostMDPClassifierContinous.h"

#include "RBFBasedQFunction.h"
#include "RBFStateModifier.h"


#include "cstate.h"
#include "cstateproperties.h"
#include "clinearfafeaturecalculator.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	// -----------------------------------------------------------------------

	AdaBoostMDPClassifierContinous::AdaBoostMDPClassifierContinous(const nor_utils::Args& args, int verbose, DataReader* datareader, int classNum, int discState )
	: CEnvironmentModel(classNum, discState), _args(args), _verbose(verbose), _classNum(classNum), _data(datareader), _incrementalReward(false), _lastReward(0.0) //CEnvironmentModel(classNum+1,classNum)
	{
		// set the dim of state space
		for( int i=0; i<_classNum;++i)
		{
			properties->setMinValue(i, 0.0);
			properties->setMaxValue(i, 1.0);
		}
		
		_exampleResult = NULL;
		
		// open result file
		string tmpFname = _args.getValue<string>("traintestmdp", 4);			
		_outputStream.open( tmpFname.c_str() );
		
		_sumAlpha = _data->getSumOfAlphas();
        
        cout << "[+] Sum of alphas: " << _sumAlpha << endl;
        
		_classifierUsed.resize(_data->getIterationNumber());
		
		if (args.hasArgument("rewards"))
		{
			double rew = args.getValue<double>("rewards", 0);
			setSuccessReward(rew); // classified correctly
			
			// the reward we incur if we use a classifier		
			rew = args.getValue<double>("rewards", 1);
			setClassificationReward( rew );
			
			rew = args.getValue<double>("rewards", 2);
			setSkipReward( rew );		
			
			setJumpReward(0.0);
		} else {
			cout << "No rewards are given!" << endl;
			exit(-1);
		}
		
		
		if (args.hasArgument("succrewartdtype"))
		{
			string succRewardMode = args.getValue<string>("succrewartdtype");
			
			if ( succRewardMode == "hamming" )
				_succRewardMode = RT_HAMMING;
			else if ( succRewardMode == "exp" )
				_succRewardMode = RT_EXP;
            else if ( succRewardMode == "logit")
                _succRewardMode = RT_LOGIT;
			else
			{
				cerr << "ERROR: Unrecognized (succrewartdtype) --succes rewards option!!" << endl;
				exit(1);
			}						
		} else {		
			_succRewardMode = RT_HAMMING;
		}
		
        if (args.hasArgument("incrementalrewardQ")) {
            _incrementalReward = true;
        }

        // for budgetted classification
        _budgetedClassification = false;
        if (args.hasArgument("budgeted"))
        {
            _budgetedClassification = true;
            cout << "[+] Budgeted Classification" << endl;
        }

        string featureCostFile;
        _featureCosts.clear();
        _featuresEvaluated.clear();

        properties->setDiscreteStateSize(0, datareader->getIterationNumber()+1);
        if (discState > 1) properties->setDiscreteStateSize(1, pow(3, (datareader->getIterationNumber() + 1)) + 1);
        
        if (args.hasArgument("featurecosts")) {
                        
            featureCostFile = args.getValue<string>("featurecosts", 0);
        
            ifstream ifs(featureCostFile.c_str());
            if (! ifs) {
                cout << "Error: could not open the feature cost file <" << featureCostFile << ">" << endl;
                exit(1);
            }
            
            AlphaReal cost;
            while (ifs >> cost) {
                _featureCosts.push_back(cost);
            }
            
            assert(_featureCosts.size() == _data->getNumAttributes());
            
            if (verbose > 2) {
                
                const NameMap& namemap = _data->getAttributeNameMap();
                cout << "[+] Feature Budget:" << endl;
                
                for (int i = 0 ; i < _featureCosts.size(); ++i) {
                    cout << "--> " << namemap.getNameFromIdx(i) << "\t" << _featureCosts[i]  << endl;
                }
            }
            
            // init the vector of evaluated features
            _featuresEvaluated.resize(_featureCosts.size(), false);
            
            properties->setDiscreteStateSize(0, (datareader->getIterationNumber() * 2)+1);
        }

        // for binary
        _positiveLabelIndex = 0;
        if ( args.hasArgument("positivelabel") )
		{
			args.getValue("positivelabel", 0, _positiveLabelName);
            const NameMap& namemap = datareader->getClassMap();
            _positiveLabelIndex = namemap.getIdxFromName( _positiveLabelName );
            
        }
        else
        {
            if (classNum < 3) {
                cout << "Error: the positive label must be given with --positivelabel" << endl;
                exit(1);
            }
        }

        _currentKeyIndex = 0;
        
        // "winners" state var init
        _currentWinnerIndex = 0;
        
//        vector<int> winners(2, 0); winners[1] = 1 ; // = { _exampleResult->getWinner(0).first, _exampleResult->getWinner(1).first }; //{-1, -1}
//        _winnersIndices[winners] = _currentWinnerIndex++;
        
        int k = 0;
        const int numClasses = datareader->getClassNumber();
        for (int i = 0; i < numClasses; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                if (i != j)
                {
                    vector<int> w(2); w[0] = i; w[1] = j;
                    _winnersIndices[w] = k++;
                }
            }
        }
        
        _proportionalAlphaNorm = false;
        _currentSumAlpha = 0.;
	}
    
    // -----------------------------------------------------------------------------------
    
    int AdaBoostMDPClassifierContinous::getPositiveLabelIndex() {
        return _positiveLabelIndex;
    }

	// -----------------------------------------------------------------------
    
    double AdaBoostMDPClassifierContinous::getClassificationCost() {
        if (_featureCosts.size() == 0) {
            return (AlphaReal)_classifierNumber;
        }
        else
        {
            AlphaReal cost = 0.;
            for (int i = 0; i < _featureCosts.size(); ++i) {
                if (_featuresEvaluated[i]) cost += _featureCosts[i];
            }
            return cost;
        }
    }

	// -----------------------------------------------------------------------

	void AdaBoostMDPClassifierContinous::getState(CState *state)
	{
		// initializes the state object
		CEnvironmentModel::getState ( state );
		
		
		// not necessary since we do not store any additional information
		
		vector<AlphaReal>& currVotesVector = _exampleResult->getVotesVector();
		
        // set the continuous state var
		if (_classNum<=0) //2
        {
			state->setNumActiveContinuousStates(1);
            double st = ((currVotesVector[_positiveLabelIndex] /_sumAlpha)+1)/2.0; // rescale between [0,1]
            state->setContinuousState(0, st);
        }
		else
        {
			state->setNumActiveContinuousStates(_classNum);
            // Set the internal state variables
            for ( int i=0; i<_classNum; ++i) {
                
                double st = 0.0;
                
                AlphaReal alphaSum = _sumAlpha;
                
                if (_proportionalAlphaNorm && !nor_utils::is_zero( _currentSumAlpha)) {
                    alphaSum = _currentSumAlpha;
                }

                st = ((currVotesVector[i] /alphaSum)+1)/2.0; // rescale between [0,1]
                state->setContinuousState(i, st);
            }
        }
		
        //  set the discrete state var
        if (_budgetedClassification) {
            int idxBias = 0;
            if (_featuresEvaluated[_currentClassifier])
                idxBias = 1;
            state->setDiscreteState(0, (_currentClassifier * 2) + idxBias);

        }
        else
            state->setDiscreteState(0, _currentClassifier);
        
        state->setDiscreteState(1, _keysIndices[_classifiersOutput]);

//        cout << "+++[DEBUG] _keysIndices[_classifiersOutput] " << _keysIndices[_classifiersOutput] << endl;
//        for (const auto & myTmpKey : _keysIndices)
//        {
//            for (const auto & myTmpKey2 : myTmpKey.first) cout << myTmpKey2 << " ";
//            cout << " -> " << myTmpKey.second << endl;
//        }
        
        vector<int> winners(2);
        winners[0] = _exampleResult->getWinner(0).first;
        winners[1] = _exampleResult->getWinner(1).first;
        
        state->setDiscreteState(2, _winnersIndices[winners]);
        
//        cout << "+++[DEBUG] _keysIndices[_classifiersOutput] " << _keysIndices[_classifiersOutput] << endl;
//        cout << "+++[DEBUG] _classifiersOutput ";
//        for (auto & i : _classifiersOutput) cout << i << " "; cout << endl;
	}

	// -----------------------------------------------------------------------
	
    void AdaBoostMDPClassifierContinous::doNextState(CPrimitiveAction *act)
	{
		CAdaBoostAction* action = dynamic_cast<CAdaBoostAction*>(act);
		
		int mode = action->getMode();
		//cout << mode << endl;
		if ( mode == 0 ) // skip
		{
            _currentSumAlpha += _data->getAlpha(_currentClassifier);

			_currentClassifier++;
            _classifiersOutput.push_back(0);
            

            KeyIndicesType::const_iterator kIt = _keysIndices.find(_classifiersOutput);
            if (kIt == _keysIndices.end()) {
                _keysIndices[_classifiersOutput] = _currentKeyIndex++;
            }

		}
		else if (mode == 1 ) // classify
		{
            
            _currentSumAlpha += _data->getAlpha(_currentClassifier);

            vector<int> votes = _data->classifyKthWeakLearner(_currentClassifier,_currentRandomInstance,_exampleResult);
            
            int classifierOutput = votes[_positiveLabelIndex];
            
            if (_featuresEvaluated.size() != 0) {
                set<int> usedCols = _data->getUsedColumns(_currentClassifier);
                for (set<int>::iterator it = usedCols.begin(); it != usedCols.end() ; ++it) {
                    _featuresEvaluated[*it] = true;
                }
            }
            
			_classifierUsed[_currentClassifier] = true;
            _classifiersOutput.push_back(classifierOutput);
			_classifierNumber++;
			_currentClassifier++;
            
            KeyIndicesType::const_iterator kIt = _keysIndices.find(_classifiersOutput);
            if (kIt == _keysIndices.end()) {
                _keysIndices[_classifiersOutput] = _currentKeyIndex++;
            }

//            vector<int> winners = { _exampleResult->getWinner(0).first, _exampleResult->getWinner(1).first };
//            map<vector<int>, int>::const_iterator wIt = _winnersIndices.find(winners);
//            if (wIt == _winnersIndices.end()) {
//                _winnersIndices[winners] = _currentWinnerIndex++;
//            }

			
		} else if (mode == 2 ) // jump to end
		{
//            _currentClassifier++;
			_currentClassifier = _data->getIterationNumber();
            
            _classifiersOutput.push_back(2);
            
            KeyIndicesType::const_iterator kIt = _keysIndices.find(_classifiersOutput);
            if (kIt == _keysIndices.end()) {
                _keysIndices[_classifiersOutput] = _currentKeyIndex++;
            }

		}
        		
		if ( _currentClassifier == _data->getIterationNumber() ) // check whether there is any weak classifier
		{
			reset = true;
			if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult ) )
			{
				failed = false;
			} else {
				failed = true;
			}
		}
	}
	
    // -----------------------------------------------------------------------
	
    void AdaBoostMDPClassifierContinous::doResetModel()
	{
		//_currentRandomInstance = (int) (rand() % _data->getNumExamples() );
		_currentClassifier = 0;
		_classifierNumber = 0;				
		_currentSumAlpha = 0.0;
		
		if (_exampleResult==NULL) 
			_exampleResult = new ExampleResults(_currentRandomInstance,_data->getClassNumber());		
		
		vector<AlphaReal>& currVotesVector = _exampleResult->getVotesVector();
		
		fill( currVotesVector.begin(), currVotesVector.end(), 0.0 );		
		fill( _classifierUsed.begin(), _classifierUsed.end(), false );
        _classifiersOutput.clear();
        
        fill( _featuresEvaluated.begin(), _featuresEvaluated.end(), false );

			
	}
    
	// -----------------------------------------------------------------------

	double AdaBoostMDPClassifierContinous::getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState)
    {
		double rew = 0.0;
		CAdaBoostAction* gridAction = dynamic_cast<CAdaBoostAction*>(action);
		int mode = gridAction->getMode();
		
		if ( _currentClassifier < _data->getIterationNumber() )
		{
			if (mode==0)
			{			
				rew = _skipReward;
			} else if ( mode == 1 )
			{
                AlphaReal whypCost = 1.;
                if (_budgetedClassification) {
                    whypCost = 0.;
                    set<int> usedCols = _data->getUsedColumns(_currentClassifier);
                    for (set<int>::iterator it = usedCols.begin(); it != usedCols.end() ; ++it) {
                        whypCost += _featureCosts[*it];
                    }
                }
                
				rew = _classificationReward * whypCost;
                
                if (_incrementalReward) {
                    
                    rew -= _lastReward;
                    if (_succRewardMode==RT_HAMMING)
                    {
                        
                        if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult )  ) // classified correctly
                        {
                            _lastReward = _successReward;// /100.0;
                        }
                    }
                    else if (_succRewardMode==RT_EXP)
                    {
                        double exploss;
                        if (_classifierNumber>0)
                        {
                            exploss = _data->getExponentialLoss( _currentRandomInstance,  _exampleResult );
                            _lastReward = 1/exploss;
                        }
                    }
                    else if (_succRewardMode==RT_LOGIT)
                    {
                        double logitloss;
                        if (_classifierNumber>0)
                        {
                            logitloss = _data->getLogisticLoss( _currentRandomInstance,  _exampleResult );
                            _lastReward = logitloss;
                        }
                    }
                    
                    rew += _lastReward;
                }

			} else if ( mode == 2 )
			{
				rew = _jumpReward;
//                {
//                    
//                    if (_succRewardMode==RT_HAMMING)
//                    {
//                        if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult )  ) // classified correctly
//                        {
//                            rew += _successReward;// /100.0;
//                        } else
//                        {
//                            rew -= _successReward;
//                        }
//                    } else if (_succRewardMode==RT_EXP)
//                    {
//                        // since the AdaBoost minimize the margin e(-y_i f(x_i)
//                        // we will maximize -1/e(y_i * f(x_i)
//                        double exploss;
//                        if (_classifierNumber>0)
//                        {
//                            exploss = _data->getExponentialLoss( _currentRandomInstance,  _exampleResult );
//                            rew += 1/exploss;
//                        }
//                        else
//                        {
//                            //exploss = exp(_data->getSumOfAlphas());
//                            //rew -= _successReward;
//                        }
//                        
//                        /*
//                         cout << "Instance index: " << _currentRandomInstance << " ";
//                         bool clRes =  _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult );
//                         if (clRes)
//                         cout << "[+] exploss: " << exploss << endl << flush;
//                         else
//                         cout << "[-] exploss: " << exploss << endl << flush;
//                         */
//                        
//                        
//                        
//                    }
//                }
			}
			
		} else {		
			
			if (_succRewardMode==RT_HAMMING)
			{//                AlphaReal margin = _data->getMargin(_currentRandomInstance, _exampleResult);
//                rew += margin;
				if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult )  ) // classified correctly
				{
					failed = false;
					rew += _successReward;// /100.0;
//                    assert(margin > 0);
				}
                else
				{
					failed = true;
					rew += -_successReward;
//					assert(margin <= 0);
                    
				}
			} else if (_succRewardMode==RT_EXP)
			{
				// since the AdaBoost minimize the margin e(-y_i f(x_i)
				// we will maximize -1/e(y_i * f(x_i)
				double exploss;
				if (_classifierNumber>0)
				{
					exploss = _data->getExponentialLoss( _currentRandomInstance,  _exampleResult );
					rew += -exploss + 1000;
				}
				else
				{
					//exploss = exp(_data->getSumOfAlphas());
					//rew -= _successReward;
				}
				
				/*
				 cout << "Instance index: " << _currentRandomInstance << " ";
				 bool clRes =  _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult );
				 if (clRes)
				 cout << "[+] exploss: " << exploss << endl << flush;
				 else
				 cout << "[-] exploss: " << exploss << endl << flush;
				 */
				
				
				
			}
            else if (_succRewardMode==RT_LOGIT)
            {
                double logitloss;
                if (_classifierNumber>0)
                {
                    logitloss = _data->getLogisticLoss( _currentRandomInstance,  _exampleResult );
                    rew += logitloss;
                }
            }
            
            else {
				cout << "Unknown succes reward type!!! Maybe it is not implemented! " << endl;
				exit(-1);
			}
		}
		return rew;
	}
    
	// -----------------------------------------------------------------------
    
	bool AdaBoostMDPClassifierContinous::classifyCorrectly()
	{
		return  _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult );
	}

	// -----------------------------------------------------------------------
    
	bool AdaBoostMDPClassifierContinous::hasithLabelCurrentElement( int i )
	{
		return  _data->hasithLabel( _currentRandomInstance, i ); 
	}		
	
    // -----------------------------------------------------------------------

	CStateModifier* AdaBoostMDPClassifierContinous::getStateSpaceRBF(unsigned int partitionNumber)
	{
		// Now we can already create our RBF network
		// Therefore we will use a CRBFFeatureCalculator, our feature calculator uses both dimensions of the model state 
		// (the angel and the angular velocity) and lays a 20x20 RBF grid over the state space. For each dimension the given sigmas are used.
		// For the calculation of useful sigmas we have to consider that the CRBFFeatureCalculator always uses the 
		// normalized state representation, so the state variables are scaled to the intervall [0,1]
		
		int numClasses = _classNum;
		if (numClasses == 2) numClasses = 1;
        
		unsigned int* dimensions = new unsigned int[numClasses];
		unsigned int* partitions = new unsigned int[numClasses];
		double* offsets = new double[numClasses];
		double* sigma = new double[numClasses];
		
		for(int i=0; i<numClasses; ++i )
		{
			dimensions[i]=i;
			partitions[i]=partitionNumber;
			offsets[i]=0.0;
			sigma[i]=1.0/(2.0*partitionNumber);;
		}
		
		
		// Now we can create our Feature Calculator
		CStateModifier *rbfCalc = new CRBFFeatureCalculator(numClasses, dimensions, partitions, offsets, sigma);
        CAbstractStateDiscretizer* disc= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
        CFeatureOperatorAnd *andCalculator = new CFeatureOperatorAnd();
        andCalculator->addStateModifier(disc);
        andCalculator->addStateModifier(rbfCalc);
        
        andCalculator->initFeatureOperator();
        
		return andCalculator;
	}
	// -----------------------------------------------------------------------
    
    CStateModifier* AdaBoostMDPClassifierContinous::getStateSpace( int divNum )
	{
		// create the discretizer with the build in classes
		// create the partition arrays
		int numClasses = _classNum;
        if (numClasses == 2) numClasses = 1;
        
		double *partitions = new double[divNum-1];
		double step = 1.0/divNum;
		for(int i=0;i<divNum-1;++i)
		{
			//cout << (i+1)*step << " " << endl << flush;
			partitions[i]= (i+1)*step;
		}
		//double partitions[] = {-0.5,-0.2,0.0,0.2,0.5}; // partition for states
		//double partitions[] = {-0.2,0.0,0.2}; // partition for states
		
        
		CAbstractStateDiscretizer** disc = new CAbstractStateDiscretizer*[numClasses+1];
		
		//disc[0] = new CSingleStateDiscretizer(0,5,partitions);
		disc[0]= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		for(int l=0;l<numClasses;++l) disc[l+1] = new CSingleStateDiscretizer(0,divNum-1,partitions);
		
		// Merge the discretizers
		CDiscreteStateOperatorAnd *andCalculator = new CDiscreteStateOperatorAnd();
		
		for(int l=0;l<=numClasses;++l) andCalculator->addStateModifier(disc[l]);
		
		
		return andCalculator;
		
	}
    
    // -----------------------------------------------------------------------------------
    
    CStateModifier* AdaBoostMDPClassifierContinous::getStateSpaceExp( int divNum, int e )
	{
		// create the discretizer with the build in classes
		// create the partition arrays
		int numClasses = getNumClasses();
		double *partitions = new double[divNum-1];
		for(int i=1;i<divNum;++i)
		{
			partitions[i-1] = pow(i,e)/pow(divNum,e);
			partitions[i-1] = (partitions[i-1]/2.0);
			cout << partitions[i-1] << " ";
		}
		cout << endl << flush;
		
		double* realPartitions = new double[(divNum-1)*2+1];
		realPartitions[divNum-1]=0.5;
		for(int i=0;i<divNum-1;++i)
		{
			realPartitions[i]= 0.5 - partitions[divNum-2-i];
		}
		for(int i=0;i<divNum-1;++i)
		{
			realPartitions[divNum+i]= 0.5 + partitions[i];
		}
		
		for(int i=0; i<(divNum-1)*2+1; ++i)
		{
			cout << realPartitions[i] << " ";
		}
		cout << endl;
		
		//double partitions[] = {-0.5,-0.2,0.0,0.2,0.5}; // partition for states
		//double partitions[] = {-0.2,0.0,0.2}; // partition for states
		
		CAbstractStateDiscretizer** disc = new CAbstractStateDiscretizer*[numClasses+1];
		
		//disc[0] = new CSingleStateDiscretizer(0,5,partitions);
		disc[0]= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		for(int l=0;l<numClasses;++l) disc[l+1] = new CSingleStateDiscretizer(0,(divNum-1)*2+1,realPartitions);
		
		// Merge the discretizers
		CDiscreteStateOperatorAnd *andCalculator = new CDiscreteStateOperatorAnd();
		
		for(int l=0;l<=numClasses;++l) andCalculator->addStateModifier(disc[l]);
		
		
		return andCalculator;
		
	}


	// -----------------------------------------------------------------------
	CStateModifier* AdaBoostMDPClassifierContinous::getStateSpaceTileCoding(unsigned int partitionNumber)
	{
		// Now we can already create our RBF network
		// Therefore we will use a CRBFFeatureCalculator, our feature calculator uses both dimensions of the model state 
		// (the angel and the angular velocity) and lays a 20x20 RBF grid over the state space. For each dimension the given sigmas are used.
		// For the calculation of useful sigmas we have to consider that the CRBFFeatureCalculator always uses the 
		// normalized state representation, so the state variables are scaled to the intervall [0,1]
		
		int numClasses = getNumClasses();
		
		unsigned int* dimensions = new unsigned int[numClasses];
		unsigned int* partitions = new unsigned int[numClasses];
		double* offsets = new double[numClasses];
		double* sigma = new double[numClasses];
		
		for(int i=0; i<numClasses; ++i )
		{
			dimensions[i]=i;
			partitions[i]=partitionNumber;
			offsets[i]=0.0;
			sigma[i]=0.025;
		}
		
		
		// Now we can create our Feature Calculator
		CStateModifier *tileCodeCalc = new CTilingFeatureCalculator(numClasses, dimensions, partitions, offsets );	
		return tileCodeCalc;
	}	

    
    // -----------------------------------------------------------------------
    
    CStateModifier* AdaBoostMDPClassifierContinous::getStateSpaceForGSBNFQFunction( int numOfFeatures){
        int numClasses = _classNum;
//        if (numClasses == 2) numClasses = 1;
        
        int multipleDescrete = 1;
        if (_budgetedClassification) {
            multipleDescrete = 2;
        }
        
		CStateModifier* retVal = new RBFStateModifier(numOfFeatures, numClasses, (_data->getIterationNumber() * multipleDescrete) +1 );
		return retVal;
        
    }
    
	// -----------------------------------------------------------------------

	void AdaBoostMDPClassifierContinous::getHistory( vector<bool>& history )
	{
		history.resize( _classifierUsed.size() );
		copy( _classifierUsed.begin(), _classifierUsed.end(), history.begin() );
	}

	// -----------------------------------------------------------------------
    
	void AdaBoostMDPClassifierContinous::getHistory( vector<int>& history )
	{
		history.clear();

        for (int i = 0; i < _classifierUsed.size(); ++i) {
            if (_classifierUsed[i] != false)
                history.push_back(i);
        }
	}

    // -----------------------------------------------------------------------
	
	void AdaBoostMDPClassifierContinous::getClassifiersOutput( vector<int>& classifiersOutput )
	{
		classifiersOutput.resize( _classifiersOutput.size() );
		copy( _classifiersOutput.begin(), _classifiersOutput.end(), classifiersOutput.begin() );
	}

    // -----------------------------------------------------------------------

	void AdaBoostMDPClassifierContinous::getCurrentExmapleResult( vector<double>& result )
	{
		vector<AlphaReal>& currVotesVector = _exampleResult->getVotesVector();
		result.resize(currVotesVector.size());
		copy( currVotesVector.begin(), currVotesVector.end(), result.begin() );
		
		for( int i =0; i<currVotesVector.size(); ++i )
		{
			result[i] = currVotesVector[i]/_sumAlpha;
		}
		
	}

    // -----------------------------------------------------------------------------------
    
    void AdaBoostMDPClassifierContinous::outHeader()
    {
//        _outputStream << setiosflags(ios::fixed);
        
        _outputStream << "ep" << "\t" <<  "full" << "\t" << "prop" << "\t" << "acc" << "\t" << "eval" << "\t" << "rwd" << "\t" << "cost" << setprecision(4) <<  endl ;

//        if (_classNum <= 2) {
//            _outputStream << "ep" << "\t" <<  "full" << "\t" << "prop" << "\t" << "err" << "\t" << "eval" << "\t" << "rwd" << "\t" << "tpr" << "\t" << "tnr" << "\t" << "cost" << setprecision(4) <<  endl ;
//        }
//        else {
//            _outputStream << "Ep" << "\t" <<  "AdaB" << "\t" << "Acc" << "\t" << "AvgEv" << "\t" << "AvgRwd" << endl << setprecision(4) ;
//        }
    }
    
	// -----------------------------------------------------------------------
        
//    void AdaBoostMDPClassifierContinous::outPutStatistic(int ep, double acc, double curracc, double uc, double sumrew )
//	{
//		_outputStream << ep << "\t" << acc << "\t" << curracc << "\t" << uc << "\t" << sumrew << endl << flush;
//	}

	// -----------------------------------------------------------------------
        
    void AdaBoostMDPClassifierContinous::outPutStatistic( BinaryResultStruct& bres )
    {
        //		_outputStream << bres.iterNumber << " " <<  bres.adaboostPerf << " " << bres.err << " " << bres.usedClassifierAvg << " " << bres.avgReward << " " << bres.TP << " " << bres.TN << " " << bres.negNumEval <<  endl;
//        _outputStream << bres.iterNumber << "\t" << 100*(1 - bres.adaboostPerf) << "\t"  <<  100*(1 - bres.itError) << "\t" << 100*(1 - bres.err) << "\t" << bres.usedClassifierAvg << "\t" << bres.avgReward << "\t" << bres.TP << "\t" << bres.TN << "\t" << bres.classificationCost <<  endl;
        _outputStream << bres.iterNumber << "\t" << 100*(1 - bres.adaboostPerf) << "\t"  <<  100*(1 - bres.itError) << "\t" << 100*(1 - bres.err) << "\t" << bres.usedClassifierAvg << "\t" << bres.avgReward << "\t" << bres.classificationCost <<  endl;

    }


} // end of namespace MultiBoost