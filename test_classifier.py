from unittest import TestCase

class Test(TestCase):
    def test_get_classifier(self):
        """
        Tests the get_classifier function
        """
        from classifier import getClassifier
        with self.assertRaises(Exception): getClassifier(10)    # testing non string value
        with self.assertRaises(Exception): getClassifier(1.0)
        with self.assertRaises(Exception): getClassifier("testString")  # testing string values not in list
        try:                                                        # testing all allowed values to see that they work
            getClassifier('VOTING')
        except Exception:
            self.fail("getClassifier() raised ExceptionType unexpectedly!")
        try:
            getClassifier('voting')
        except Exception:
            self.fail("getClassifier() raised ExceptionType unexpectedly!")
        try:
            getClassifier('sgb')
        except Exception:
            self.fail("getClassifier() raised ExceptionType unexpectedly!")
        try:
            getClassifier('ada')
        except Exception:
            self.fail("getClassifier() raised ExceptionType unexpectedly!")
        try:
            getClassifier('rf')
        except Exception:
            self.fail("getClassifier() raised ExceptionType unexpectedly!")
        try:
            getClassifier('bag')
        except Exception:
            self.fail("getClassifier() raised ExceptionType unexpectedly!")

    def test_sgb(self):
        """
        Tests the sgb function
        """
        from classifier import sgb
        with self.assertRaises(Exception): sgb("test")  # tests non integer values
        with self.assertRaises(Exception): sgb(1.0)
        with self.assertRaises(Exception): sgb(-10)     # tests negative integer values
        try:
            sgb(10)
        except Exception:
            self.fail("sgb() raised ExceptionType unexpectedly!")


    def test_ada_boost(self):
        """
        Tests the adaboost function
        """
        from classifier import adaBoost
        with self.assertRaises(Exception): adaBoost("test") # tests non integer values
        with self.assertRaises(Exception): adaBoost(1.0)
        with self.assertRaises(Exception): adaBoost(-10) # tests negative integer values
        try:
            adaBoost(10)
        except Exception:
            self.fail("sgb() raised ExceptionType unexpectedly!")

    def test_random_forest(self):
        """
        Tests the random forest function
        """
        from classifier import randomForest
        with self.assertRaises(Exception): randomForest("test") # tests non integer values
        with self.assertRaises(Exception): randomForest(1.0)
        with self.assertRaises(Exception): randomForest(-10) # tests negative integer values

        try:
            randomForest(10)
        except Exception:
            self.fail("random_forest() raised ExceptionType unexpectedly!")

    def test_bagging(self):
        """
        Tests the bagging function
        """
        from classifier import bagging
        with self.assertRaises(Exception): bagging("test") # tests non integer values
        with self.assertRaises(Exception): bagging(1.0)
        with self.assertRaises(Exception): bagging(-10) # tests negative integer values

        try:
            bagging(10)
        except Exception:
            self.fail("bagging() raised ExceptionType unexpectedly!")

    def test_result_validation(self):
        """
        Tests the result validation function
        """
        from classifier import resultValidation
        from sklearn.neighbors import KNeighborsClassifier
        testmodel = KNeighborsClassifier()
        with self.assertRaises(Exception): resultValidation(testmodel,[],[],"test") # tests various configurations of non valid inputs
        with self.assertRaises(Exception): resultValidation(testmodel,[],[],-1)
        with self.assertRaises(Exception): resultValidation(testmodel,[],0,1)
        with self.assertRaises(Exception): resultValidation(testmodel,[],"test",1)
        with self.assertRaises(Exception): resultValidation(testmodel,[],1.0,1)
        with self.assertRaises(Exception): resultValidation(testmodel,0,[],1)
        with self.assertRaises(Exception): resultValidation(testmodel,"test",[],1)
        with self.assertRaises(Exception): resultValidation(testmodel,1.0,[],1)
        with self.assertRaises(Exception): resultValidation(testmodel,[],[],1)
        # try:                                                                               # Tests a valid input
        #     resultValidation(testmodel,[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],1)
        # except Exception:
        #     self.fail("bagging() raised ExceptionType unexpectedly!")

    def test_get_best_features(self):
        """
        Tests the get best features function
        """
        from classifier import getBestFeatures, treeFeatures
        with self.assertRaises(Exception): getBestFeatures(treeFeatures,5,"test") # tests various configurations of non valid inputs
        with self.assertRaises(Exception): getBestFeatures(treeFeatures,"test",5)
        with self.assertRaises(Exception): getBestFeatures(treeFeatures,"test","test")
        with self.assertRaises(Exception): getBestFeatures(treeFeatures,5,-1)
        with self.assertRaises(Exception): getBestFeatures(treeFeatures,5,1.0)
        with self.assertRaises(Exception): getBestFeatures(treeFeatures,-1,5)
        with self.assertRaises(Exception): getBestFeatures(treeFeatures,1.0,5)
        try:                                                                        # Tests a valid input
            getBestFeatures(treeFeatures,5,5)
        except Exception:
            self.fail("getBestFeatures() raised ExceptionType unexpectedly!")