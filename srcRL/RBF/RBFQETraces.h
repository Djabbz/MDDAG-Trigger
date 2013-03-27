/*
 *  RBFQETraces.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __RBFQETRACES_H
#define __RBFQETRACES_H

//#define RBFDEB

#include "cqetraces.h"
#include "AdaBoostMDPClassifierAdv.h"
#include "RBFBasedQFunction.h"
#include <vector>
#include <list>

class RBFBasedQFunctionBinary;
using namespace std;


typedef vector<vector<double> > OneIterETrace;
typedef list<OneIterETrace > CustomETrace;
typedef CustomETrace::iterator ETIterator;
typedef CustomETrace::reverse_iterator ETReverseIterator;

typedef vector<vector<RBFParams> > OneIterETraceMulti;
typedef list<OneIterETraceMulti > CustomETraceMulti;
typedef CustomETraceMulti::iterator ETMIterator;
typedef CustomETraceMulti::reverse_iterator ETMReverseIterator;


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: this structure supposes that we never come back to a state. It's fine for now.
class GSBNFQETraces : public CAbstractQETraces 
{
protected:
    vector<RBFParams> _margins;
	vector<int>	 _iters;
    CustomETraceMulti _eTraces;
    int _numDimensions;
	list<int> _actions;
    
    double _learningRate;
	
    double _minActivation;
    double _maxError;
    double _maxErrorFactor;
	
	CStateCollection* _currentState;
    
    
public:
    GSBNFQETraces(CAbstractQFunction *qFunction) : CAbstractQETraces(qFunction) {
        _numDimensions = dynamic_cast<GSBNFBasedQFunction*> (qFunction)->getNumDimensions();
        _maxError = -1;
        _learningRate = qFunction->getParameter("QLearningRate");
        _maxErrorFactor = qFunction->getParameter("MaxTDErrorDivFactor");
        _minActivation =  qFunction->getParameter("MinActivation");
    }
    virtual ~GSBNFQETraces() {};
    
    virtual void updateETraces(CAction *action, CActionData *data = NULL) 
	{
        double mult = getParameter("Lambda") * getParameter("DiscountFactor");
        ETMIterator eIt = _eTraces.begin();
		list<int>::iterator actionIt = _actions.begin();
        
        vector<RBFParams>::iterator itMargin = _margins.begin();
        vector<int>::iterator itIters = _iters.begin();		
        
        while (eIt != _eTraces.end())
        {
            OneIterETraceMulti & oneItEtrace = *eIt;
            OneIterETraceMulti gradient;
            
//            dynamic_cast<GSBNFBasedQFunction* >(qFunction)->getGradient(*itMargin, *itIters, *actionIt, gradient);
            
            for (int j = 0; j < oneItEtrace.size(); ++j) {
                for (int k = 0; k < oneItEtrace[j].size(); ++k) {
                    for (int h = 0; h < _numDimensions; ++h) {
                        oneItEtrace[j][k][h] = oneItEtrace[j][k][h] * mult ;
                    }
                    
                    //                    oneItEtrace[j][k] +=  gradient[j][k];
                }
            }
            
            ++eIt;
            //++stateIt;
            ++actionIt;
            ++itMargin;
			++itIters;
        }
	}
    
    virtual void addETrace(CStateCollection *state, CAction *action, double factor = 1.0, CActionData *data = NULL) 
	{
		//CState* currentState = static_cast<CState*>(state)->clone();		
		//CState* copyState = state->getState()->clone();
		//_states.push_back( copyState );
		_currentState = state;
		//MultiBoost::CAdaBoostAction* currentAction = NULL;
		//int mode = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
		//currentAction = new MultiBoost::CAdaBoostAction( mode );
        
        int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
		_actions.push_back( actionIndex );
        
        
        
        OneIterETraceMulti gradient;
        dynamic_cast<GSBNFBasedQFunction* >(qFunction)->getGradient(state, actionIndex, gradient);
        
        _eTraces.push_back(gradient);
        
 		CState* currState = state->getState();
		int currIter = currState->getDiscreteState(0);
		_iters.push_back(currIter);

        RBFParams margin(_numDimensions);
        for (int i = 0; i < _numDimensions; ++i) {
            margin[i] = currState->getContinuousState(i);
        }
		_margins.push_back(margin);
		
	}
    
    virtual void updateQFunction(double td) 
	{
        
        double tderror = td / _learningRate;
        double activation = dynamic_cast<GSBNFBasedQFunction* >(qFunction)->getMaxActivation(_currentState, _actions.back());
        
        //int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(_actions.back())->getMode();
        bool addCenter = qFunction->getParameter("AddCenterOnError") > 0.5;
 
        if ( addCenter && (fabs(tderror) > _maxError/_maxErrorFactor) && (activation < _minActivation)) {
            //add a center
            //the sigma is calculated by the qfunction
            RBFParams& newCenter  = _margins.back();
            dynamic_cast<GSBNFBasedQFunction* >(qFunction)->addCenter(tderror, newCenter, _iters.back(), _actions.back(), _maxError);
            
            //update the etrace structure
            _eTraces.pop_back();
            
            OneIterETraceMulti gradient;
            dynamic_cast<GSBNFBasedQFunction* >(qFunction)->getGradient(newCenter, _iters.back(), _actions.back(), gradient);
            
            _eTraces.push_back(gradient);
        }
        
		//list<CStateCollection*>::iterator invitState = _states.begin();
		list<int>::iterator invitAction = _actions.begin();
        ETMIterator invitTrace = _eTraces.begin();
        
        vector<RBFParams>::iterator itMargin = _margins.begin();
        vector<int>::iterator itIters = _iters.begin();		
        
#ifdef RBFDEB				

		cout << "Action: " << _actions.rbegin() << " TD " << td << endl;
#endif
		
		for (; itMargin != _margins.end(); ++itMargin, ++itIters, ++invitAction, ++invitTrace)
		{
            OneIterETraceMulti& currentETrace = *invitTrace;
			
			
			RBFParams currMargin = *itMargin;
			int currIter = *itIters;
			int currentAction = *invitAction;
            
//			CState* artificialState = _currentState->getState()->clone();
//            
//            for (int i = 0; i < _numDimensions; ++i) {
//                artificialState->setContinuousState(i, currMargin[i]);
//            }
//            artificialState->setDiscreteState(0, currIter);
//			
//#ifdef RBFDEB					
//			cout << "(A:" << *invitAction->getMode() << "," << flush;    			
//			cout << "O:" << dynamic_cast<GSBNFBasedQFunction* >(qFunction)->getValue(artificialState, *invitAction)<< "," << flush;
//#endif			
//			dynamic_cast<GSBNFBasedQFunction* >(qFunction)->updateValue(artificialState, currentAction, td, currentETrace);
            dynamic_cast<GSBNFBasedQFunction* >(qFunction)->updateValue(currIter, currMargin, currentAction, td, currentETrace);
//#ifdef RBFDEB					
//			cout << "N:" << dynamic_cast<GSBNFBasedQFunction* >(qFunction)->getValue(artificialState, *invitAction) << ")" << endl << flush;
//#endif			
//			delete artificialState;
		}
#ifdef RBFDEB				
		cout << endl;
		if (index==2) cout << "End of episode" << endl;
#endif		
	}	
    
    virtual void resetETraces() 
	{
		_actions.clear();
        
        _eTraces.clear();
		_iters.clear();
		_margins.clear();				
#ifdef RBFDEB		
		cout << "------------------------------------------------" << endl;
#endif
	}
    
};


#endif