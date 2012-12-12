/*
 *  RBFBasedQFunction.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef __RBFBASEDQFUNCTION_H
#define __RBFBASEDQFUNCTION_H

#include "cqfunction.h"
#include "cqetraces.h"
#include "cgradientfunction.h"
#include "cstatemodifier.h"
#include "RBFStateModifier.h"
#include "AdaBoostMDPClassifierAdv.h"
#include "cfeaturefunction.h"
#include <newmat/newmat.h>
#include <newmat/newmatio.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//typedef ColumnVector RBFParams;
typedef vector<double> RBFParams;

class MultiRBF {
protected:

    RBFParams _mean;
    RBFParams _sigma;
    
    RBFParams _alpha;
    string _id;
    
public:
    MultiRBF(int numDimensions) : _mean(numDimensions), _sigma(numDimensions), _alpha(numDimensions) //for simplification (tmp)
    {
        for (int i=0; i < numDimensions; ++i) {
            _mean[i] = 0.;
            _sigma[i] = 0.;
            _alpha[i] = 0.;
        }
    }
    
    MultiRBF() : _mean(0), _sigma(0), _alpha(0) 
    {    }
    
    RBFParams& getMean() { return _mean; }
	RBFParams& getSigma() { return _sigma; }
	RBFParams& getAlpha() { return _alpha; }
    string& getId() { return _id;}
    
	void setMean( RBFParams& m ) { 
        for (int i=0; i < _mean.size(); ++i) {
            _mean[i] = m[i];
        }
        
        if (_mean[0] != _mean[0]) {
            assert(false);
        }
    }
	void setMean( double m ) { 
        for (int i=0; i < _mean.size(); ++i) {
            _mean[i] = m;
        }
        if (_mean[0] != _mean[0]) {
            assert(false);
        }
    }
	void setSigma( RBFParams& s ) { 
        for (int i=0; i < _sigma.size(); ++i) {
            _sigma[i] = s[i];
        }                
    }
	void setSigma( double s ) { 
        for (int i=0; i < _sigma.size(); ++i) {
            _sigma[i] = s;
        }        
    }
    void setAlpha( RBFParams& a ) {
        for (int i=0; i < _alpha.size(); ++i) {
            _alpha[i] = a[i];
        }              
    }
    void setAlpha( double a ) { 
        for (int i=0; i < _alpha.size(); ++i) {
            _alpha[i] = a;
        }      
    }
    
	void setId( const string& id ) { _id = id; }
    
    double getActivationFactor(RBFParams& x)
    {
        assert(x.size() == _mean.size());
        
        double factor = 0.0;
        
        for (int i = 0; i < _mean.size(); ++i)
        {
            factor += (x[i] - _mean[i])*(x[i] - _mean[i]) / (_sigma[i]*_sigma[i]);	
        }
        factor = - factor / 2;
        return exp(factor);
        
    }
    
    double getValue(RBFParams& x) {
        return _alpha[0] * getActivationFactor(x);
    }
    
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GSBNFBasedQFunction : public CAbstractQFunction//: public RBFBasedQFunctionBinary 
{
protected:
    int _featureNumber;	
	CActionSet* _actions;
	int _numberOfActions;
    int _numberOfIterations;
    
    double _muAlpha;
    double _muMean;
    double _muSigma;
    
    vector<double> _bias;
    
    double _maxSigma;
    
    int _numDimensions;
    vector<vector<vector<MultiRBF> > > 	_rbfs; // <action, iteration, rbf number>
public:
    GSBNFBasedQFunction(CActionSet *actions, CStateModifier* statemodifier );
    
    virtual double getMuAlpha() { return _muAlpha; }
	virtual double getMuMean() { return _muMean; }
	virtual double getMuSigma() { return _muSigma; }
	
	virtual void setMuMean( double m ) { _muMean=m; }
	virtual void setMuSigma( double s ) { _muSigma = s; }
	virtual void setMuAlpha( double a ) { _muAlpha = a; }
    
    int getNumDimensions() {
        return _numDimensions;
    }
    void uniformInit(double* init);
    void setBias(vector<double>& bias);
    double getValue(CStateCollection *state, CAction *action, CActionData *data);
    double getMaxActivation(CStateCollection *state, int action, CActionData *data=NULL);
    void getActivationFactors(RBFParams& margin, int currIter, int action, vector<double>& factors);
    void addCenter(double tderror, RBFParams& newCenter, int iter, int action, double& maxError);
    void updateValue(int currIter, RBFParams& margin, int action, double td, vector<vector<RBFParams> >& eTraces);
    void getGradient(CStateCollection *state, int action, vector<vector<RBFParams> >& gradient);
    void getGradient(RBFParams& margin, int currIter, int action, vector<vector<RBFParams> >& gradient);
    void saveActionValueTable(FILE* stream, int dim=0);
    vector<int> saveCentersNumber(FILE* stream);
    void loadQFunction(const string& fileName);
    CAbstractQETraces* getStandardETraces();
};

#endif