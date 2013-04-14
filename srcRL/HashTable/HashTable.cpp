//#include "RBFQETraces.h"
#include "HashTable.h"
#include "HashTableStateModifier.h"

//#include "RBFStateModifier.h"
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//CAbstractQETraces* GSBNFBasedQFunction::getStandardETraces()
//{
//    return new GSBNFQETraces(this);
//}
//


// -----------------------------------------------------------------------------------

HashTable::HashTable(CActionSet *actions, CStateModifier* sm,  MultiBoost::AdaBoostMDPClassifierContinous* classifier, int numDim) : CAbstractQFunction(actions)
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

double HashTable::getValue(CStateCollection* state, CAction *action, CActionData *data) {
    int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
    ValueKey key;
    MDDAGState mddagState(state->getState());
    getKey(mddagState, key);
    AlphaReal value;
    getTableValue(actionIndex, key, value);
    return value;
};

// -----------------------------------------------------------------------------------

bool HashTable::getTableValue(int actionIndex, ValueKey& key, AlphaReal& outValue, AlphaReal defaultValue)
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

void HashTable::getKey(MDDAGState& state, ValueKey& key)
{    
    //        vector<int> history;
    //        _classifier->getHistory( history );
    //        _classifier->getClassifiersOutput(history);
    
    //        const size_t numDimensions = 0;// currState->getNumActiveContinuousStates();
    size_t numDimensions = state.continuousStates.size();

    _numDimensions = 0;
    numDimensions = _numDimensions;
    
    size_t numEvaluations ;
    
    int keyIdx = state.discreteStates[1];
    vector<int> history = _classifier->getHistoryFromState(keyIdx);
    
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
        if (*it == 0.) history.pop_back();
        else break;
    }
    
    //        cout << "+++[DEBUG] history.size " << history.size() << endl;
    //        numEvaluations = history.size() == 0 ? 0 : history.size() - 1; // minus one to delete the last weakhyp evaluated //history.size() < 2 ? history.size() : 2 ;
    numEvaluations = history.size();
//    numEvaluations = 0;
    
    key.clear();
    key.resize(numDimensions + numEvaluations);//+ 1
    
    int i = 0;
//    key[i++] = state.discreteStates[0];
    
    for (int j = 0; j < numDimensions; ++i, ++j) {
        
        // rounding operation
        //            int score = (int)(currState->getContinuousState(j) * 1000);
        //            key[i] = score;
        key[i] = state.continuousStates[j];
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

double HashTable::getMaxValue(MDDAGState& state)
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

void HashTable::addTableEntry(double tderror, MDDAGState& state, int actionIndex)
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

void HashTable::updateValue(MDDAGState& state, CAction *action, double td, CActionData * actionData)
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

void HashTable::saveActionValueTable(FILE* stream, int dim)
{
    //        fprintf(stream, "Q-Hash Table\n");
    
    ValueTableType::iterator tableIt = _valueTable.begin();
    
    for (; tableIt != _valueTable.end(); ++tableIt) {
        
        ValueKey key = tableIt->first;
        vector<AlphaReal> values = tableIt->second;
        
        fprintf(stream, "( ");
        ValueKey::iterator keyIt = key.begin();
        if (keyIt != key.end()) fprintf(stream, "%d ", (int)*(keyIt++));
        
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

