/*
 *  RBFBasedQFunction.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef __HASHTABLE_H
#define __HASHTABLE_H

#include <map>
#include <vector>
#include <cmath> //for round
#include <algorithm>

#include "Defaults.h"

#include "cqfunction.h"
#include "cqetraces.h"
#include "cgradientfunction.h"
#include "cstatemodifier.h"
#include "HashTableStateModifier.h"
#include "AdaBoostMDPClassifierAdv.h"
//#include "cfeaturefunction.h"
#include <newmat/newmat.h>
#include <newmat/newmatio.h>

#include "HashTableETraces.h"
#include "AdaBoostMDPClassifierContinous.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef vector<double> ValueKey;
typedef map<ValueKey, vector<AlphaReal> > ValueTableType;

class RBFStateModifier;

class HashTable : public CAbstractQFunction//: public RBFBasedQFunctionBinary
{
protected:
    
    ValueTableType                      _valueTable;
	CActionSet*                         _actions;
	int                                 _numberOfActions;
    AlphaReal                           _learningRate;
    CStateModifier*                     _stateProperties;
    int                                 _numDimensions;
    MultiBoost::AdaBoostMDPClassifierContinous*     _classifier;
        
public:
    
    // -----------------------------------------------------------------------------------
    
    HashTable(CActionSet *actions, CStateModifier* sm,  MultiBoost::AdaBoostMDPClassifierContinous* classifier, int numDim) : CAbstractQFunction(actions)
    {
        _stateProperties = sm;
        _classifier = classifier;
        _actions = actions;
        _numberOfActions = (int)actions->size();
        
        addParameter("QLearningRate", 0.2);

        _numDimensions = numDim;
        if (_numDimensions == 2) --_numDimensions;
        
    }

    // -----------------------------------------------------------------------------------
	
    /**
     * Return the value for a given action and a given score
     */
    double getValue(CStateCollection *state, CAction *action, CActionData *data) {
        int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();        
        ValueKey key;
        getKey(state, key);
        AlphaReal value;
        getTableValue(actionIndex, key, value);
        return value;
    };
    
    // -----------------------------------------------------------------------------------
    
    /**
     * Helper function to return the state value
     * or a default value if the state has never
     * been visited.
     */
    bool getTableValue(int actionIndex, ValueKey& key, AlphaReal& outValue, AlphaReal defaultValue = 0.)
    {
        ValueTableType::const_iterator it = _valueTable.find(key);
        if (it == _valueTable.end()) {
            outValue = defaultValue;
            return false;
        }
        else {
            outValue = it->second[actionIndex];
            return true;
        }
    }
    
    // -----------------------------------------------------------------------------------
    
    /**
     * Build a key for the hash table
     * from the current state
     */
    void getKey(CStateCollection *state, ValueKey& key)
    {
        CState* currState = state->getState();
        
        
        vector<int> history;
//        _classifier->getHistory( history );
//        _classifier->getClassifiersOutput(history);

//        const size_t numDimensions = 0;// currState->getNumActiveContinuousStates();
        const size_t numDimensions = currState->getNumActiveContinuousStates();
        size_t numEvaluations ;
        
//        numEvaluations = history.size() == 0 ? 0 : history.size() - 1; // minus one to delete the last weakhyp evaluated //history.size() < 2 ? history.size() : 2 ;
        numEvaluations = history.size();

        key.clear();
        key.resize(numDimensions + numEvaluations + 1);//+ 1
        
        int i = 0;
        key[i++] = currState->getDiscreteState(0);

        for (int j = 0; j < numDimensions; ++i, ++j) {
            
            // rounding operation
            int score = (int)(currState->getContinuousState(j) * 1000);
//            key[i] = score;
            key[i] = currState->getContinuousState(j);
        }
        
//        vector<int>::reverse_iterator rIt = history.rbegin();
        
        for (int k = 0; k < numEvaluations; ++i, ++k) { //,++rIt
            key[i] = history[k]; //*rIt
        }
        

        
//        cout << "+++[DEBUG] curr " << currState->getDiscreteState(0) << endl;
//        if (history.size()) cout << "+++[DEBUG] first " << history[0] << endl;
//        char c; cin >> c;
    }
    
    // -----------------------------------------------------------------------------------
    
    void setLearningRate(AlphaReal r) { _learningRate = r; }
    
    // -----------------------------------------------------------------------------------
    
    
    /**
     * Return the max value for a
     * given score
     */
    double getMaxValue(CStateCollection *state)
    {
        ValueKey key;
        getKey(state, key);
        
        AlphaReal maxVal = 0.0;
        
        for( int i=0; i < _numberOfActions; ++i )
        {
            AlphaReal value;
            getTableValue(i, key, value);
            if (value > maxVal) {
                maxVal = value;
            }
        }
        return maxVal;
        cout << endl << "max activation " << maxVal << endl;
    }
    
    // -----------------------------------------------------------------------------------
    
    void addTableEntry(double tderror, CStateCollection *state, int actionIndex)
    {
        ValueKey key;
        getKey(state, key);
        
//        cout << "+++[DEBUG] new entry: " ;
//        for (int i = 0; i < key.size(); ++i) {
//            cout << key[i] << flush;
//        }
//        cout << endl;
        
//        cout << "+++[DEBUG] New entry: " ;
//        for (int i = 0; i < key.size(); ++i) {
//            cout << key[i] << " ";
//        }
//        cout << endl;
        
        ValueTableType::const_iterator it = _valueTable.find(key);
        assert(it == _valueTable.end());

        _valueTable[key].resize(_numberOfActions);
        _valueTable[key][actionIndex] = tderror;
    }
    
    // -----------------------------------------------------------------------------------
    
    void updateValue(CStateCollection *state, CAction *action, double td, CActionData * = NULL)
    {
        if (td != td) {
            assert(false);
        }

        int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
        
        ValueKey key;
        getKey(state, key);
        
        AlphaReal value;
        bool entryExists = getTableValue(actionIndex, key, value);
        
        if (entryExists)
            _valueTable[key][actionIndex] += td; //* _learningRate;
        else
            addTableEntry(td, state, actionIndex);
    };
    
    // -----------------------------------------------------------------------------------
    
    void saveActionValueTable(FILE* stream, int dim=0)
    {
//        fprintf(stream, "Q-Hash Table\n");

        ValueTableType::iterator tableIt = _valueTable.begin();
        
        for (; tableIt != _valueTable.end(); ++tableIt) {
            
            ValueKey key = tableIt->first;
            vector<AlphaReal> values = tableIt->second;
            
            fprintf(stream, "( ");
            ValueKey::iterator keyIt = key.begin();
            fprintf(stream, "%d ", (int)*(keyIt++));
            
            //TMP
            for (int d = 0; d < _numDimensions; ++d, ++keyIt) {
                fprintf(stream, "%f ", ((*keyIt)*2) - 1);
            }

            for (; keyIt != key.end(); ++keyIt) {
                fprintf(stream, "%d ", (int)(*keyIt));
            }

            fprintf(stream, ")\t");
            
            for (int i = 0; i < values.size(); ++i) {
                fprintf(stream, "%f ", values[i]);
            }
            
            fprintf(stream, "\n");            
        }

    }

    // -----------------------------------------------------------------------------------
    
    CAbstractQETraces* getStandardETraces() { return new HashTableETraces(this, _stateProperties); } ;
    
    // -----------------------------------------------------------------------------------
    
    //    void getGradient(CStateCollection *state, int action, vector<vector<RBFParams> >& gradient);
    //    void getGradient(RBFParams& margin, int currIter, int action, vector<vector<RBFParams> >& gradient);
    //    void loadQFunction(const string& fileName);

};

#endif