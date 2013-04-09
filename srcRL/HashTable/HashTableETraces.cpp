#include "cqetraces.h"
#include "cstatecollection.h"

#include "AdaBoostMDPClassifierAdv.h"
#include "HashTableETraces.h"

#include "HashTable.h"

#include <vector>
#include <list>

using namespace std;

void HashTableETraces::updateETraces(CAction *action, CActionData *data)
{
    double mult = getParameter("Lambda") * getParameter("DiscountFactor");
    list<double>::iterator eIt = _eTraces.begin();
    
//    DebugPrint('e',"\n[+] Update e-traces. Factor: %f\n --> [ ", mult);
    while (eIt != _eTraces.end())
    {
        (*eIt) *= mult;
//        DebugPrint('e',"%f ", *eIt);
        ++eIt;
    }
//    DebugPrint('e',"]\n");
}

// -----------------------------------------------------------------------------------

void HashTableETraces::addETrace(CStateCollection *state, CAction *action, double factor, CActionData *data) 
{
//		_currentState = state;

    CState* currState = state->getState();
    int it = currState->getDiscreteState(0);

//    DebugPrint('e',"\n[+] Add e-trace to state : %d and action %d", it, dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode());

    _eTraces.push_back(factor);
    
    _eTraceStates.push_back( MDDAGState( currState ));
//        _eTraceStates->addStateCollection(state);
    
//        int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
    _actions.push_back( action );
}

// -----------------------------------------------------------------------------------

void HashTableETraces::updateQFunction(double td)
{
//    DebugPrint('e',"\n[+] Q-Function update\n");
    if (td == 0.)
        return;
    list<CAction*>::iterator itAction = _actions.begin();
    list<double>::iterator itTrace = _eTraces.begin();
    list<MDDAGState>::iterator itState = _eTraceStates.begin();
    
    for (; itTrace != _eTraces.end(); ++itTrace, ++itAction, ++itState)
    {
//        int it = (*itState).discreteStates[0];
//        DebugPrint('e'," --> Update state : %d \t with %f * %f = %f\n", it, (*itTrace), td, (*itTrace)*td);
        
        if ((*itTrace) != 0.) dynamic_cast<HashTable*>(qFunction)->updateValue(*itState, *itAction, (*itTrace)*td);
    }
}

// -----------------------------------------------------------------------------------

void HashTableETraces::resetETraces()
{
    _eTraces.clear();
    _actions.clear();
    _eTraceStates.clear();
//        _eTraceStates->clearStateLists();
}

// -----------------------------------------------------------------------------------
