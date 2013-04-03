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
#include "HashTable.h"
#include <vector>
#include <list>

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: this structure supposes that we never come back to a state. It's fine for now.
class HashTableETraces: public CAbstractQETraces
{
protected:
//    vector<RBFParams> _margins;
//	vector<int>	 _iters;
//    CustomETraceMulti _eTraces;
//    int _numDimensions;

	list<CAction*>                  _actions;
//    double                      _learningRate;
//	CStateCollection*           _currentState;
    
//    CStateCollectionList*       _eTraceStates;
    list<CStateCollection*>         _eTraceStates;
    list<double>                    _eTraces;
    
    
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
        
//        DebugPrint('e',"\nUpdate e-traces. Factor: %f\n[", mult);
        while (eIt != _eTraces.end())
        {
//            DebugPrint('e',"%f ", *eIt);
            (*eIt) *= mult;
            ++eIt;
        }
//        DebugPrint('e',"]\n");
	}
    
    // -----------------------------------------------------------------------------------
    
    virtual void addETrace(CStateCollection *state, CAction *action, double factor = 1.0, CActionData *data = NULL) 
	{
//		_currentState = state;
        _eTraces.push_back(factor);
        _eTraceStates.push_back(state);
//        _eTraceStates->addStateCollection(state);
        
//        int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
		_actions.push_back( action );
	}
    
    // -----------------------------------------------------------------------------------
    
    virtual void updateQFunction(double td) 
	{
        double tderror = td; // / _learningRate;
  
//        dynamic_cast<HashTable * >(qFunction)->addTableEntry(tderror, _eTraceStates.back(), _actions.back());
        
		list<CAction*>::iterator itAction = _actions.begin();
        list<double>::iterator itTrace = _eTraces.begin();
        list<CStateCollection*>::iterator itState = _eTraceStates.begin();
        
		for (; itTrace != _eTraces.end(); ++itTrace, ++itAction, ++itState)
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