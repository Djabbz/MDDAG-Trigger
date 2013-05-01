#include "RBFBasedQFunction.h"
#include "RBFQETraces.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GSBNFBasedQFunction::GSBNFBasedQFunction(CActionSet *actions, CStateModifier* statemodifier )
: CAbstractQFunction(actions)
{
    RBFStateModifier* smodifier = dynamic_cast<RBFStateModifier*>( statemodifier );

    
    const int iterationNumber = smodifier->getNumOfIterations();
    const int featureNumber = smodifier->getNumOfRBFsPerIteration();
    const int numOfClasses = smodifier->getNumOfClasses();
    
    _featureNumber = featureNumber;
    _numberOfIterations = iterationNumber;
    _numDimensions = numOfClasses;
    
    if (_numDimensions == 2) {
        _numDimensions--;
    }
    
    _actions = actions;
    _numberOfActions = actions->size();
    
    //init the bias
    _bias.resize(_numberOfActions, 0.0);
    
    _rbfs.clear();
    _rbfs.resize(_numberOfActions);
    
//    CActionSet::iterator it=_actions->begin();
    for (int ac=0; ac < _numberOfActions; ++ac) 
    {
//        int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(*it)->getMode();
        int actionIndex = ac;
        _rbfs[actionIndex].resize( iterationNumber );
        for( int i=0; i<iterationNumber; ++i)
        {
            _rbfs[actionIndex][i].reserve(_featureNumber);
            for (int j = 0; j < _featureNumber; ++j) {
                MultiRBF rbf(_numDimensions);
                _rbfs[actionIndex][i].push_back(rbf);
            }
        }
    }
    
    addParameter("InitRBFSigma", 0.01);
    addParameter("AddCenterOnError", 1.);
    addParameter("NormalizedRBFs", 1);
    addParameter("MaxTDErrorDivFactor", 10);
    addParameter("MinActivation", 0.3);
    addParameter("QLearningRate", 0.2);
    addParameter("MaxRBFNumber", 1000);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GSBNFBasedQFunction::setBias(vector<double>& bias)
{
//    assert(bias.size() == _numberOfActions);
    for (int i = 0; i < _numberOfActions; ++i) {
        _bias[i] = bias[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GSBNFBasedQFunction::uniformInit(double* init)
{
    double initSigma = getParameter("InitRBFSigma");
    
    CActionSet::iterator it=_actions->begin();
    for(;it!=_actions->end(); ++it )
    {	
        int index = dynamic_cast<MultiBoost::CAdaBoostAction*>(*it)->getMode();
        
        double initAlpha = 0;
        if (init != NULL) {
            //warning  : no check on the bounds of init                
            initAlpha = init[index];
        }
                
        int iterationNumber = _rbfs[index].size();
        for( int i=0; i<iterationNumber; ++i)
        {                
            int numFeat = _rbfs[index][i].size();

            double sigma = 1./ (2.2*numFeat);
            
            if (sigma > initSigma) {
                sigma = initSigma;
            }
            
            for (int j = 0; j < numFeat; ++j) {
                //                    if (numFeat % 2 == 0) {
                _rbfs[index][i][j].setMean((j+1) * 1./(numFeat+1));
                //                    }
                //                    else {
                //                        _rbfs[*it][i][j].setMean(j * 1./numFeat);   
                //                    }
                
                _rbfs[index][i][j].setAlpha(initAlpha);

                _rbfs[index][i][j].setSigma(sigma);
                
                stringstream tmpString("");
                tmpString << "[ac_" << index << "|it_" << i << "|fn_" << j << "]";
                _rbfs[index][i][j].setId( tmpString.str() );
            }
        }
    }           
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


double GSBNFBasedQFunction::getValue(CStateCollection *state, CAction *action, CActionData *data)
{
    CState* currState = state->getState();
    int currIter = currState->getDiscreteState(0);
    
    RBFParams margin(_numDimensions);
    for (int i = 0; i < _numDimensions; ++i) {
        margin[i] = currState->getContinuousState(i);
    }
    
    int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
    
    vector<MultiRBF>& currRBFs = _rbfs[actionIndex][currIter];		
    double retVal = 0.0;
    double rbfSum = 0.0;
    double bias = _bias[actionIndex];
    
    if (currRBFs.size() == 0) {
        return  bias;
    }
    
    for( int i=0; i<currRBFs.size(); ++i )
    {
        if (rbfSum != rbfSum) {
            assert(false);
        }
        
        rbfSum += currRBFs[i].getActivationFactor(margin);
        retVal += currRBFs[i].getValue(margin);
    }		

    bool norm  = getParameter("NormalizedRBFs") > 0.5;
    if (norm) {
        retVal /= rbfSum;
    }
    
    assert( retVal == retVal);
    return retVal;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double GSBNFBasedQFunction::getMaxActivation(CStateCollection *state, int action, CActionData *data )
{
    CState* currState = state->getState();
    int currIter = currState->getDiscreteState(0);

    RBFParams margin(_numDimensions);
    for (int i = 0; i < _numDimensions; ++i) {
        margin[i] = currState->getContinuousState(i);
    }
    
    double maxVal = 0.0;
    
    vector<MultiRBF>& currRBFs = _rbfs[action][currIter];		
    for( int i=0; i<currRBFs.size(); ++i )
    {
        double act = currRBFs[i].getActivationFactor(margin); 
        if (act > maxVal) {
            maxVal = act;
        }
    }		
    return maxVal;
    cout << endl << "max activation " << maxVal << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GSBNFBasedQFunction::getActivationFactors(RBFParams& margin, int currIter, int action, vector<double>& factors)
{
    double sum = 0.0;
    
    vector<MultiRBF>& currRBFs = _rbfs[action][currIter];		
    factors.clear();
    
    if (currRBFs.size() == 0) {
        return ;
    }

    factors.resize(currRBFs.size());
                   
    for( int i=0; i<currRBFs.size(); ++i )
    {
        double af = currRBFs[i].getActivationFactor(margin);
        factors[i] = af;
        sum += af;
    }
     
    bool norm  = getParameter("NormalizedRBFs") > 0.5;
    if (norm) {
        for( int i=0; i<currRBFs.size(); ++i )
        {
            factors[i] /= sum;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GSBNFBasedQFunction::addCenter(double tderror, RBFParams& newCenter, int iter, int action, double& maxError)
{
    vector<MultiRBF>& rbfs = _rbfs[action][iter];

    if (rbfs.size() >= getParameter("MaxRBFNumber")) {
        return;
    }
    
    double newSigma = getParameter("InitRBFSigma");

    MultiRBF newRBF(_numDimensions);
    newRBF.setMean(newCenter);
    newRBF.setAlpha(tderror);
    newRBF.setSigma(newSigma);
    
    stringstream tmpString("");
//    tmpString << "[ac_" << action << "|it_" << iter << "|fn_" << index << "]";
    newRBF.setId( tmpString.str() );
    
#ifdef RBFDEB
    cout << "New center : " << newRBF.getId() << " at " << newCenter[0] << endl;
    cout << tderror << "\t" << newCenter[0] << "\t" << newSigma << endl ;
#endif
    
    rbfs.push_back( newRBF );
    
    for (int i=0; i < rbfs.size(); ++i) {
        if (rbfs[i].getAlpha()[0] > maxError) {
            maxError = rbfs[i].getAlpha()[0];
        }
    }
    
    //normalizeNetwork();
    
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GSBNFBasedQFunction::updateValue(int currIter, RBFParams& margin, int action, double td, vector<vector<RBFParams> >& eTraces)
{
    
    if (td != td) {
        assert(false);
    }
    
//    CState* currState = state->getState();
//    int currIter = currState->getDiscreteState(0);
//
//    RBFParams margin(_numDimensions);
//    for (int i = 0; i < _numDimensions; ++i) {
//        margin[i] = currState->getContinuousState(i);
//    }
//    assert (_rbfs.find(action) != _rbfs.end());
    
    vector<MultiRBF>& rbfs = _rbfs[action][currIter];
    int numCenters = rbfs.size();//_rbfs[currIter][action].size();
    
    for (int i = 0; i < numCenters; ++i) {
        
        RBFParams& alpha = rbfs[i].getAlpha();
        RBFParams& mean = rbfs[i].getMean();
        RBFParams& sigma = rbfs[i].getSigma();
        
//        cout << "RBF : " << rbfs[i].getId() << endl;
//        cout << alpha  << mean  << sigma << endl;

        //update the center and shape
        vector<RBFParams>& currentGradient = eTraces[i];
        
        RBFParams newAlpha(_numDimensions);
        RBFParams newMean(_numDimensions);
        RBFParams newSigma(_numDimensions);
        
        for (int j = 0; j < _numDimensions; ++j) {
            newAlpha[j] = alpha[j] + _muAlpha * currentGradient[0][j] * td  ;
            newMean[j] = mean[j] + _muMean * currentGradient[1][j] * td  ;
            newSigma[j] = sigma[j] + _muSigma * currentGradient[2][j] * td  ;
            
            if (newMean[j] != newMean[j]) {
                assert(false);
            }
            
            if (newSigma[j] != newSigma[j]) {
                cout << currentGradient[2][j] << endl;
                assert(false);
            }
            
        }
        
        rbfs[i].setAlpha(newAlpha);
        rbfs[i].setMean(newMean);
        rbfs[i].setSigma(newSigma);
#ifdef RBFDEB			
        cout << rbfs[i].getID() << " ";
#endif
    }		
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GSBNFBasedQFunction::getGradient(CStateCollection *state, int action, vector<vector<RBFParams> >& gradient)
{
    CState* currState = state->getState();
    int currIter = currState->getDiscreteState(0);

    RBFParams margin(_numDimensions);
    for (int i = 0; i < _numDimensions; ++i) {
        margin[i] = currState->getContinuousState(i);
    }
    
    getGradient(margin, currIter, action, gradient);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GSBNFBasedQFunction::getGradient(RBFParams& margin, int currIter, int action, vector<vector<RBFParams> >& gradient)
{

//tmp
//    for (int ac=0; ac < _numberOfActions; ++ac) 
//    {
//        for( int i=0; i<_numberOfIterations; ++i)
//        {
//            cout << _rbfs[ac][i].size() << " ";
//        }
//        cout << endl ;
//    }
    
    vector<MultiRBF>& rbfs = _rbfs[action][currIter];
    int numCenters = rbfs.size();
    gradient.clear();
    gradient.resize(numCenters);

    vector<double> activationFactors;
    getActivationFactors(margin, currIter, action, activationFactors);
    
    for (int i = 0; i < numCenters; ++i) {
        RBFParams& alpha = rbfs[i].getAlpha();
        RBFParams& mean = rbfs[i].getMean();
        RBFParams& sigma = rbfs[i].getSigma();
        
        RBFParams distance(_numDimensions);
        for (int j = 0; j < _numDimensions; ++j) {
            distance[j] = margin[j] - mean[j];
        }
        
        double rbfValue = activationFactors[i];
        
        RBFParams alphaGrad = RBFParams(_numDimensions);
        for (int j = 0; j < _numDimensions; ++j) {
            alphaGrad[j] = rbfValue;
        }
        
        
        RBFParams meanGrad(_numDimensions);
        RBFParams sigmaGrad(_numDimensions);
        for (int j = 0; j < _numDimensions; ++j) {
//            meanGrad[j] = rbfValue * alpha[0] * distance[j] / (sigma[j]*sigma[j]);
//            sigmaGrad[j] = rbfValue * alpha[0] * distance[j] * distance[j]/ (sigma[j] * sigma[j] * sigma[j]);

            meanGrad[j] = rbfValue * alpha[0] * distance[j] * distance[j] / (sigma[j]*sigma[j]);
            sigmaGrad[j] = rbfValue * alpha[0] * distance[j] * distance[j]/ (sigma[j]);
            
            if (meanGrad[j] != meanGrad[j]) {
                assert(false);
            }
            
            if (sigmaGrad[j] != sigmaGrad[j]) {
                assert(false);
            }
            
            bool norm  = getParameter("NormalizedRBFs") > 0.5;
            if (norm) {
                meanGrad[j] *= (1 - rbfValue);
                sigmaGrad[j] *= (1 - rbfValue);
            }
        }

//        RBFParams meanGrad = SP( SP((1 - rbfValue) , SP(rbfValue, SP(alpha,distance))) , 1/SP(sigma,sigma));
//        RBFParams sigmaGrad = SP( SP((1 - rbfValue) , SP(rbfValue, SP(alpha,SP(distance,distance)))) , 1/SP(sigma, SP(sigmasigma)));
        
        gradient[i].resize(3);
        gradient[i][0] = alphaGrad;
        gradient[i][1] = meanGrad;
        gradient[i][2] = sigmaGrad;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GSBNFBasedQFunction::saveActionValueTable(FILE* stream, int dim)
{
    fprintf(stream, "Q-FeatureActionValue Table\n");
//    CActionSet::iterator it;
    
    for (int j = 0; j < _numberOfIterations; ++j) {
        for (int k = 0; k < _numberOfActions; ++k) {
            fprintf(stream,"classifier %d action %d: ", j,k);
            for (int i = 0; i < _rbfs[k][j].size(); ++i) {
                fprintf(stream,"%f %f %f ", _rbfs[k][j][i].getAlpha()[dim], _rbfs[k][j][i].getMean()[dim], _rbfs[k][j][i].getSigma()[dim]);
            }
            fprintf(stream, "\n");
        }
        
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<int> GSBNFBasedQFunction::saveCentersNumber(FILE* stream)
{
    fprintf(stream, "GSNBF centers number\n");
    //    CActionSet::iterator it;
    
    vector<int> maxCenterNumber(_numberOfActions);
    
    for (int j = 0; j < _numberOfIterations; ++j) {
        fprintf(stream,"classifier %d:", j);
        for (int k = 0; k < _numberOfActions; ++k) {
            fprintf(stream," %d", _rbfs[k][j].size());
            if (maxCenterNumber[k] < _rbfs[k][j].size()) {
                maxCenterNumber[k] = _rbfs[k][j].size();
            }
        }
        fprintf(stream, "\n");        
    }
    return maxCenterNumber;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GSBNFBasedQFunction::loadQFunction(const string& fileName)
{
    ifstream inFile(fileName.c_str());
    if (!inFile.is_open())
    {
        cerr << "ERROR: Cannot open Q Table file <" << fileName << ">!" << endl;
        exit(1);
    }
    
    _rbfs.clear();
    _rbfs.resize(_numberOfActions);
    for (int j = 0; j < _numberOfActions; ++j) {
        _rbfs[j].resize(_numberOfIterations);
    }
    
    nor_utils::StreamTokenizer coarseST(inFile, "\n\r\t");
    string tmp = coarseST.next_token();
    
    int i;
    for (i = 0; i < _numberOfIterations && coarseST.has_token(); ++i) {
        for (int ac = 0; ac < _numberOfActions; ++ac) {
            string coarseToken = coarseST.next_token();
            stringstream ss(coarseToken);
            nor_utils::StreamTokenizer fineST(ss, " \n\r\t");
            
            if (fineST.next_token().compare("classifier") == 0) {
                for (int j=0; j<3; ++j) { //eliminate the first useless words
                    string tmp = fineST.next_token();
                }
                
                while (fineST.has_token()) {

                    string tmp = fineST.next_token();
                    if (tmp.empty()) 
                        continue;
                    
                    stringstream alphaSS(tmp);
                    stringstream meanSS(fineST.next_token());
                    stringstream sigmaSS(fineST.next_token());
                    
                    double alpha;
                    alphaSS >> alpha;
                    
                    double sigma;
                    sigmaSS >> sigma;
                    
                    double mean;
                    meanSS >> mean;
                    
                    MultiRBF rbf(_numDimensions);;
                    
                    rbf.setAlpha(alpha);
                    rbf.setMean(mean);
                    rbf.setSigma(sigma);
                    
                    _rbfs[ac][i].push_back(rbf);
                }
            }    
        }
    }
    
    if (i < _numberOfIterations) {
        cout << "Warning: the number of weak learners loaded is less than that in the QTable file.\n";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CAbstractQETraces* GSBNFBasedQFunction::getStandardETraces()
{
    return new GSBNFQETraces(this);
}

