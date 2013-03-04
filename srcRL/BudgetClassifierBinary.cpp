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
				rew = _classificationReward;
                
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
                    //                    _lastReward = rew;
                }
                
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
        
		state->setDiscreteState(0, _currentClassifier);
	}

}