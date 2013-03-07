//
//  BudgetClassifierBinary.cpp
//  MDDAG
//
//  Created by ∂jabbz on 04/03/13.
//  Copyright (c) 2013 AppStat. All rights reserved.
//

#include "BudgetClassifierBinary.h"

using namespace std;

namespace MultiBoost {
    
    // -----------------------------------------------------------------------------------
    
    BudgetClassifierBinary::BudgetClassifierBinary( const nor_utils::Args& args, int verbose, DataReader* datareader, string featureCostFile) : AdaBoostMDPClassifierContinousBinary(args, verbose, datareader)
    {
        ifstream ifs(featureCostFile.c_str());
        if (! ifs) {
            cout << "Error: could not open the feature cost file <" << featureCostFile << ">" << endl;
            exit(1);
        }
        
        _featureCosts.clear();
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
        _featuresEvaluated.clear();
        _featuresEvaluated.resize(_featureCosts.size(), false);
        
        properties->setDiscreteStateSize(0, (datareader->getIterationNumber() * 2)+1);
        
        // init the properties of the state
        //            delete properties;
        //            properties = new CStateProperties(1, 2);
        //
        //            properties->setDiscreteStateSize(0, datareader->getIterationNumber()+1); // the current base classifier
        //            properties->setDiscreteStateSize(1, 2);
        //			properties->setMinValue(0, 0.0);
        //			properties->setMaxValue(0, 1.0);
    }

    // -----------------------------------------------------------------------------------
    
    void BudgetClassifierBinary::doNextState(CPrimitiveAction *act)
	{
		CAdaBoostAction* action = dynamic_cast<CAdaBoostAction*>(act);
		
		int mode = action->getMode();
		//cout << mode << endl;
		if ( mode == 0 ) // skip
		{
			_currentClassifier++;
		}
		else if (mode == 1 ) // classify
		{
            double classifierOutput = _data->classifyKthWeakLearner(_currentClassifier,_currentRandomInstance,_exampleResult);
            
            set<int> usedCols = _data->getUsedColumns(_currentClassifier);
            for (set<int>::iterator it = usedCols.begin(); it != usedCols.end() ; ++it) {
                _featuresEvaluated[*it] = true;
//                _classifierNumber += _featureCosts[*it];
            }
            
			_currentSumAlpha += classifierOutput;
			_classifierUsed[_currentClassifier]=true;
            _classifiersOutput.push_back(classifierOutput);
			_classifierNumber++;
			_currentClassifier++;
            

		} else if (mode == 2 ) // jump to end
		{
			_currentClassifier = _data->getIterationNumber();
		}
		
		
		if ( _currentClassifier == _data->getIterationNumber() ) // check whether there is any weak classifier
		{
			reset = true;
			if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult ) )
			{
				failed = true;
			} else {
				failed = false;
			}
		}
	}

	// -----------------------------------------------------------------------

	void BudgetClassifierBinary::doResetModel()
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
        
        fill( _featuresEvaluated.begin(), _featuresEvaluated.end(), false );
        _classifiersOutput.clear();
        
	}
    
    // -----------------------------------------------------------------------------------
    
    void BudgetClassifierBinary::getState(CState *state)
	{
        // initializes the state object
		CEnvironmentModel::getState ( state );
		
		
		// not necessary since we do not store any additional information
		
        state->setNumActiveContinuousStates(1);
		
		// a reference for clarity and speed
		vector<AlphaReal>& currVotesVector = _exampleResult->getVotesVector();
        
        double st = ((currVotesVector[_positiveLabelIndex] /_sumAlpha)+1)/2.0; // rescale between [0,1]
        state->setContinuousState(_positiveLabelIndex, st);
        
        
        int idxBias = 0;
        if (_featuresEvaluated[_currentClassifier])
            idxBias = 1;
		state->setDiscreteState(0, (_currentClassifier * 2) + idxBias);
	}

    // -----------------------------------------------------------------------------------
    
    double BudgetClassifierBinary::getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState) {
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
                AlphaReal whypCost = 0.;
                set<int> usedCols = _data->getUsedColumns(_currentClassifier);
                for (set<int>::iterator it = usedCols.begin(); it != usedCols.end() ; ++it) {
                    whypCost += _featureCosts[*it];
                }

				rew = _classificationReward * whypCost;
                
//                if (_incrementalReward) {
//                    
//                    rew -= _lastReward;
//                    if (_succRewardMode==RT_HAMMING)
//                    {
//                        if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult )  ) // classified correctly
//                        {
//                            _lastReward = _successReward;// /100.0;
//                        }
//                    }
//                    else if (_succRewardMode==RT_EXP)
//                    {
//                        double exploss;
//                        if (_classifierNumber>0)
//                        {
//                            exploss = _data->getExponentialLoss( _currentRandomInstance,  _exampleResult );
//                            _lastReward = 1/exploss;
//                        }
//                    }
//                    else if (_succRewardMode==RT_LOGIT)
//                    {
//                        double logitloss;
//                        if (_classifierNumber>0)
//                        {
//                            logitloss = _data->getLogisticLoss( _currentRandomInstance,  _exampleResult );
//                            _lastReward = logitloss;
//                        }
//                    }
//                    
//                    
//                    rew += _lastReward;
//                    //                    _lastReward = rew;
//                }
                
			} else if ( mode == 2 )
			{
				rew = _jumpReward;
			}
			
		} else {
			if (_verbose>3)
			{
				// restore somehow the history
				//cout << "Get the history(sequence of actions in this episode)" << endl;
				//cout << "Size of action history: " << _history.size() << endl;
			}
			
            //            rew -= _lastReward;
			
			if (_succRewardMode==RT_HAMMING)
			{
				if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult )  ) // classified correctly
				{
					failed = false;
					if (hasithLabelCurrentElement(_positiveLabelIndex))
						rew += _successReward;// /100.0;
					else //is a negative element
						rew += _successReward;
				} else
				{
					failed = true;
					//rew += -_successReward;
                    if (hasithLabelCurrentElement(_positiveLabelIndex))
                        rew += _failOnPositivesPenalty;
                    else
                        rew += _failOnNegativesPenalty;
				}
			} else if (_succRewardMode==RT_EXP)
			{
				// since the AdaBoost minimize the margin e(-y_i f(x_i)
				// we will maximize -1/e(y_i * f(x_i)
				double exploss;
				if (_classifierNumber>0)
				{
					exploss = _data->getExponentialLoss( _currentRandomInstance,  _exampleResult );
					rew += 1/exploss;
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
    
    // -----------------------------------------------------------------------------------
    
    void BudgetClassifierBinary::outPutStatistic( BinaryResultStruct& bres ) {
        _outputStream << bres.iterNumber << "\t" <<  bres.origAcc << "\t" << bres.acc << "\t" << bres.usedClassifierAvg << "\t" << bres.avgReward << "\t" << bres.TP << "\t" << bres.TN << "\t" << bres.negNumEval <<  endl;

    }
    
    // -----------------------------------------------------------------------------------
    
    AlphaReal BudgetClassifierBinary::getClassificationCost()
    {
        AlphaReal cost = 0.;
        for (int i = 0; i < _featureCosts.size(); ++i)
            if (_featuresEvaluated[i]) cost += _featureCosts[i];

        return cost;
    }
    
    // -----------------------------------------------------------------------------------
    
}