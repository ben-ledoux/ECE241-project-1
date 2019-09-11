"""
UMass ECE 241 - Advanced Programming
Project #1     Fall 2018
SongLibrary.py - SongLibrary class
"""

from Song import Song
import random
import time
import csv
#Needed to plot cdf
#import numpy as np
#from pylab import *

class TreeNode:
    def __init__(self, key, val, left=None, right=None, parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent
        self.balanceFactor = 0


    def hasLeftChild(self):
        return self.leftChild


    def hasRightChild(self):
        return self.rightChild


    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self


    def isRightChild(self):
        return self.parent and self.parent.rightChild == self


    def isRoot(self):
        return not self.parent


    def isLeaf(self):
        return not (self.rightChild or self.leftChild)


    def hasAnyChildren(self):
        return self.rightChild or self.leftChild


    def hasBothChildren(self):
        return self.rightChild and self.leftChild


    def replaceNodeData(self, key, value, lc, rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self

class BinarySearchTree:

    def __init__(self):
        self.root = None
        self.size = 0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def put(self, key, val):
        if self.root:
            self._put(key, val, self.root)
        else:
            self.root = TreeNode(key, val)
        self.size = self.size + 1

    def _put(self, key, val, currentNode):
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                self._put(key, val, currentNode.leftChild)
            else:
                currentNode.leftChild = TreeNode(key, val, parent=currentNode)
                self.updateBalance(currentNode.leftChild)
        else:
            if currentNode.hasRightChild():
                self._put(key, val, currentNode.rightChild)
            else:
                currentNode.rightChild = TreeNode(key, val, parent=currentNode)
                self.updateBalance(currentNode.rightChild)

    def updateBalance(self, node):
        #print(1)
        if node.balanceFactor > 1 or node.balanceFactor < -1:
            self.rebalance(node)
            return

        if node.parent != None:
            if node.isLeftChild():
                node.parent.balanceFactor += 1
            elif node.isRightChild():
                node.parent.balanceFactor -= 1

            if node.parent.balanceFactor != 0:
                self.updateBalance(node.parent)

    def rebalance(self, node):
        #print(2)
        #print(node.hasLeftChild())
        #print(node.hasRightChild())
        if node.balanceFactor < 0:
            if node.rightChild.balanceFactor > 0:
                #print(11)
                self.rotateRight(node.rightChild)
                self.rotateLeft(node)
            else:
                self.rotateLeft(node)
        elif node.balanceFactor < 0:
            #print(12)
            if node.leftChild.balanceFactor < 0:
                self.rotateLeft(node.leftChild)
                self.rotateRight(node)
            else:
                self.rotateRight(node)

    def rotateLeft(self, rotRoot):
        #print(3)
        newRoot = rotRoot.rightChild
        rotRoot.rightChild = newRoot.leftChild
        if newRoot.leftChild != None:
            newRoot.leftChild.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isRoot():
            self.root = newRoot
        else:
            if rotRoot.isLeftChild():
                rotRoot.parent.leftChild = newRoot
            else:
                rotRoot.parent.rightChild = newRoot
        newRoot.leftChild = rotRoot
        rotRoot.parent = newRoot
        rotRoot.balanceFactor = rotRoot.balanceFactor + 1 - min(newRoot.balanceFactor, 0)
        newRoot.balanceFactor = newRoot.balanceFactor + 1 + max(rotRoot.balanceFactor, 0)

    def rotateRight(self, rotRoot):
        #print(4)
        newRoot = rotRoot.leftChild
        rotRoot.leftChild = newRoot.rightChild
        if newRoot.rightChild != None:
            newRoot.rightChild.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isRoot():
            self.root = newRoot
        else:
            if rotRoot.isRightChild():
                rotRoot.parent.rightChild = newRoot
            else:
                rotRoot.parent.leftChild = newRoot
        newRoot.rightChild = rotRoot
        rotRoot.parent = newRoot
        rotRoot.balanceFactor = rotRoot.balanceFactor + 1 - min(newRoot.balanceFactor, 0)
        newRoot.balanceFactor = newRoot.balanceFactor + 1 + max(rotRoot.balanceFactor, 0)

    def __setitem__(self, k, v):
        self.put(k, v)

    def get(self, key):
        if self.root:
            res = self._get(key, self.root)
            if res:
                return res.payload
            else:
                return None
        else:
            return None

    def _get(self, key, currentNode):
        if not currentNode:
            return None
        elif currentNode.key == key:
            return currentNode
        elif key < currentNode.key:
            return self._get(key, currentNode.leftChild)
        else:
            return self._get(key, currentNode.rightChild)

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        if self._get(key, self.root):
            return True
        else:
            return False

    def delete(self, key):
        if self.size > 1:
            nodeToRemove = self._get(key, self.root)
            if nodeToRemove:
                self.remove(nodeToRemove)
                self.size = self.size - 1
            else:
                raise KeyError('Error, key not in tree')
        elif self.size == 1 and self.root.key == key:
            self.root = None
            self.size = self.size - 1
        else:
            raise KeyError('Error, key not in tree')

    def __delitem__(self, key):
        self.delete(key)

    def spliceOut(self):
        if self.isLeaf():
            if self.isLeftChild():
                self.parent.leftChild = None
            else:
                self.parent.rightChild = None
        elif self.hasAnyChildren():
            if self.hasLeftChild():
                if self.isLeftChild():
                    self.parent.leftChild = self.leftChild
                else:
                    self.parent.rightChild = self.leftChild
                self.leftChild.parent = self.parent
            else:
                if self.isLeftChild():
                    self.parent.leftChild = self.rightChild
                else:
                    self.parent.rightChild = self.rightChild
                self.rightChild.parent = self.parent

    def findSuccessor(self):
        succ = None
        if self.hasRightChild():
            succ = self.rightChild.findMin()
        else:
            if self.parent:
                if self.isLeftChild():
                    succ = self.parent
                else:
                    self.parent.rightChild = None
                    succ = self.parent.findSuccessor()
                    self.parent.rightChild = self
        return succ

    def findMin(self):
        current = self
        while current.hasLeftChild():
            current = current.leftChild
        return current

    def remove(self, currentNode):
        if currentNode.isLeaf():  # leaf
            if currentNode == currentNode.parent.leftChild:
                currentNode.parent.leftChild = None
            else:
                currentNode.parent.rightChild = None
        elif currentNode.hasBothChildren():  # interior
            succ = currentNode.findSuccessor()
            succ.spliceOut()
            currentNode.key = succ.key
            currentNode.payload = succ.payload

        else:  # this node has one child
            if currentNode.hasLeftChild():
                if currentNode.isLeftChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.leftChild
                elif currentNode.isRightChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.leftChild
                else:
                    currentNode.replaceNodeData(currentNode.leftChild.key,
                                                currentNode.leftChild.payload,
                                                currentNode.leftChild.leftChild,
                                                currentNode.leftChild.rightChild)
            else:
                if currentNode.isLeftChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.rightChild
                elif currentNode.isRightChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.rightChild
                else:
                    currentNode.replaceNodeData(currentNode.rightChild.key,
                                                currentNode.rightChild.payload,
                                                currentNode.rightChild.leftChild,
                                                currentNode.rightChild.rightChild)

class SongLibrary:
    """
    Intialize your Song library here.
    You can initialize an empty songArray, empty BST and
    other attributes such as size and whether the array is sorted or not

    """

    def __init__(self):
        self.songArray = list()
        self.songBST = None
        self.isSorted = False
        self.size = 0

    """
    load your Song library from a given file. 
    It takes an inputFilename and store the songs in songArray
    """

    def loadLibrary(self, inputFilename):
        #loads song file and creates song objects from each row
        with open(inputFilename) as csv_file:
            csv_reader = list(csv.reader(csv_file, delimiter=','))
            for row in range(len(csv_reader)):
                songString = str(csv_reader[row][0]) + ',' + str(csv_reader[row][1]) + ',' + str(csv_reader[row][2]) + ',' + str(csv_reader[row][3]) + ',' + str(csv_reader[row][4])
                self.songArray.append(Song(songString))
                self.size = self.size + 1

    """
    Linear search function.
    It takes a query string and attibute name (can be 'title' or 'artist')
    and return the number of songs found in the library.
    Return -1 if no songs is found.
    Note that, Each song name is unique in the database,
    but each artist can have several songs.
    """

    def linearSearch(self, query, attribute):
        found = 0
        done = False
        i = 0
        #go through every item in the songArray
        if attribute == "artist":
            while i < self.size:
                if self.songArray[i].artist == query:
                    #because there can be multiple songs by one artist need to go through the whole
                    #array and count how many are found
                    found = found + 1
                i = i + 1

        elif attribute == "title":
            while i < self.size and not done:
                if self.songArray[i].title == query:
                    found = 1
                    #all song titles are unique so we can stop once we find a match
                    done = True
                i = i + 1
        if found == 0:
            found = -1
        return found

    """
    Build a BST from your Song library based on the song title. 
    Store the BST in songBST variable
    """
    #adds every song to the BST
    def buildBST(self):
        self.songBST = BinarySearchTree()
        i = 0
        while i < self.size:
            self.songBST.put(self.songArray[i].title, self.songArray[i])
            i = i + 1

    """
    Implement a search function for a query song (title) in the songBST.
    Return the song information string
    (After you find the song object, call the toString function)
    or None if no such song is found.
    """

    def searchBST(self, query):
        currentNode = self.songBST.root
        found = False
        while not found:
            # if song can't be found return none
           if currentNode == None:
               return None
           elif query == currentNode.key:
               found = True
               return currentNode.payload
           elif query > currentNode.key:
               if currentNode.hasRightChild:
                    currentNode = currentNode.rightChild
               #if song can't be found return none
               else:
                   found = True
                   return None
           elif query < currentNode.key:
               if currentNode.hasLeftChild:
                   currentNode = currentNode.leftChild
               else:
                   found = True
                   return None

    """
    Return song libary information
    """

    def libraryInfo(self):
        return "Size: " + str(self.size) + ";  isSorted: " + str(self.isSorted)

    """
    Sort the songArray using QuickSort algorithm based on the song title.
    The sorted array should be stored in the same songArray.
    Remember to change the isSorted variable after sorted
    """
    #quicksort implementation from class
    def partition(self, alist, first, last):
        pivotvalue = alist[first]

        leftmark = first + 1
        rightmark = last

        done = False
        while not done:
            #slight adjustment for how to call the values from the song array
            while leftmark <= rightmark and alist[leftmark].title <= pivotvalue.title:
                leftmark = leftmark + 1

            while alist[rightmark].title >= pivotvalue.title and rightmark >= leftmark:
                rightmark = rightmark - 1

            if rightmark < leftmark:
                done = True
            else:
                temp = alist[leftmark]
                alist[leftmark] = alist[rightmark]
                alist[rightmark] = temp

        temp = alist[first]
        alist[first] = alist[rightmark]
        alist[rightmark] = temp

        return rightmark

    def quickSortHelper(self, alist, first, last):
        if first < last:
            splitpoint = self.partition(alist, first, last)

            self.quickSortHelper(alist, first, splitpoint - 1)
            self.quickSortHelper(alist, splitpoint + 1, last)

    def quickSort(self):
        self.quickSortHelper(self.songArray, 0, self.size - 1)
        self.isSorted = True



# WRITE YOUR OWN TEST UNDER THAT IF YOU NEED
if __name__ == '__main__':
    songLib = SongLibrary()
    songLib.loadLibrary("TenKsongs.csv")
    songLib.quickSort()
    print(songLib.libraryInfo())


    #building array of 100 random songs
    testSongs = [] * 100
    for i in range(100):
        testSongs.append(songLib.songArray[random.randint(0, len(songLib.songArray) - 1)])

    #testing linear search time
    startTime = time.time()
    for song in testSongs:
        s = songLib.linearSearch(song.title, 'title')
    endTime = time.time()
    print(endTime - startTime)

    #time to build and search BST
    bstStart = time.time()
    songLib.buildBST()
    for song in testSongs:
        s = songLib.searchBST(song.title)
    bstEnd = time.time()
    print(bstEnd - bstStart)

    #dx = 1
    #x = np.arange(0, 1820, dx)
    #y = [0] * 1820
    #for song in songLib.songArray:
    #    d = int(float(song.duration))
    #    for i in range(1820):
    #        if d < i:
    #            y[i] = y[i] + 1

    #for i in range(1820):
    #   y[i] = float(y[i]) / songLib.size

    #plot(x, y)
    #show()





    #print(songLib.linearSearch("Roy Rogers", "artist"))
    #print(songLib.linearSearch("Mr. Goose", "title"))
    #print("sorted")
    #print(songLib.searchBST("Mr. Goose").toString())