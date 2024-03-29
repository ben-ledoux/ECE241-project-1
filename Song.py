"""
UMass ECE 241 - Advanced Programming
Project #1     Fall 2018
Song.py - Song class
"""

class Song:

    """
    Intial function for Song object.
    parse a given songRecord string to song object.
    For an example songRecord such as "0,Qing Yi Shi,Leon Lai,203.38893,5237536"
    It contains attributes (ID, title, artist, duration, trackID)
    """
    def __init__(self, songRecord):
        #splits the songRecord string based on , delimter from csv
        s = songRecord.split(',')
        self.ID = s[0]
        self.title = s[1]
        self.artist = s[2]
        self.duration = s[3]
        self.trackID = s[4]

    def getArtist(self):
        return self.artist

    def toString(self):
        return "Title: " + self.title + ";  Artist: " + self.artist


# WRITE YOUR OWN TEST UNDER THAT IF YOU NEED
if __name__ == '__main__':

    sampleSongRecord = "0,Qing Yi Shi,Leon Lai,203.38893,5237536"
    sampleSong = Song(sampleSongRecord)
    print(sampleSong.toString())