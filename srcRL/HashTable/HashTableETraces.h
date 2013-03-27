/*
 *  RBFQETraces.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __HASHTABLETRACE_H
#define __HASHTABLETRACE_H

//#define RBFDEB

#include "cqetraces.h"
#include "cstatecollection.h"

#include "AdaBoostMDPClassifierAdv.h"
#include "RBFBasedQFunction.h"
#include <vector>
#include <list>

class RBFBasedQFunctionBinary;
using namespace std;

typedef vector<double> RBFParams;

typedef vector<vector<RBFParams> > OneIterETraceMulti;
typedef list<OneIterETraceMulti > CustomETraceMulti;
typedef CustomETraceMulti::iterator ETMIterator;
typedef CustomETraceMulti::reverse_iterator ETMReverseIterator;


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: this structure supposes that we never come back to a state. It's fine for now.
class HashTableETraces: public CAbstractQETraces
{
protected:
//    vector<RBFParams> _margins;
//	vector<int>	 _iters;
//    CustomETraceMulti _eTraces;
//    int _numDimensions;

	list<CAction*>                   _actions;
//    double                      _learningRate;
//	CStateCollection*           _currentState;
    
//    CStateCollectionList*       _eTraceStates;
    list<CStateCollection*>       _eTraceStates;
    list<double>                _eTraces;
    
    
public:
    
    // -----------------------------------------------------------------------------------
    
    HashTableETraces(CAbstractQFunction *qFunction, CStateProperties *modelState) : CAbstractQETraces(qFunction) {
//        _learningRate = qFunction->getParameter("QLearningRate");
//        _eTraceStates = new CStateCollectionList(modelState);
    }
    
    // -----------------------------------------------------------------------------------
    
    virtual ~HashTableETraces() { //delete _eTraceStates;
    };
    
    // -----------------------------------------------------------------------------------
    
    virtual void updateETraces(CAction *action, CActionData *data = NULL) 
	{
        double mult = getParameter("Lambda") * getParameter("DiscountFactor");
        list<double>::iterator eIt = _eTraces.begin();
        
        while (eIt != _eTraces.end())
        {
            *eIt *= mult;
            ++eIt;
        }
	}
    
    // -----------------------------------------------------------------------------------
    
    virtual void addETrace(CStateCollection *state, CAction *action, double factor = 1.0, CActionData *data = NULL) 
	{
//		_currentState = state;
        _eTraces.push_back(1.);
        _eTraceStates.push_back(state);
//        _eTraceStates->addStateCollection(state);
        
//        int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
		_actions.push_back( action );
	}
    
    // -----------------------------------------------------------------------------------
    
    virtual void updateQFunction(double td) 
	{
        
        double tderror = td; // / _learningRate;
        
		list<CAction*>::iterator itAction = _actions.begin();
        list<double>::iterator itTrace = _eTraces.begin();
        list<CStateCollection*>::iterator itState = _eTraceStates.begin();
        int state = 0;
        
		for (; itTrace != _eTraces.end(); ++itTrace, ++itAction, ++state)
		{
//            _eTraceStates->getStateCollection(_eTraceStates->getNumStateCollections() - state, buffState);
            
            
            //dynamic_cast<HashTable * >
            (qFunction)->updateValue(*itState, *itAction, (*itTrace)*td);
		}
	}
    
    // -----------------------------------------------------------------------------------
    
    virtual void resetETraces() 
	{
	    _eTraces.clear();
        _actions.clear();
        _eTraceStates.clear();
//        _eTraceStates->clearStateLists();
	}
    
    // -----------------------------------------------------------------------------------
    
};


#endif