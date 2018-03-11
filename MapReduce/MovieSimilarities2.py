# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 10:09:19 2018

@author: faith
"""

from mrjob.job import MRJob
from mrjob.step import MRStep
from io import open


class MovieSimilarities2(MRJob):

    def configure_args(self):
        super(MovieSimilarities2, self).configure_args()
        self.add_file_arg('--items', help='Path to u.item')

    def load_movie_names(self):
        # Load database of movie names.
        self.movieNames = {}

        with open("u.item", encoding='ascii', errors='ignore') as f:
            for line in f:
                fields = line.split('|')
                self.movieNames[int(fields[0])] = fields[1]

    def steps(self):
        return [
            MRStep(mapper=self.mapper_parse_input,
                   reducer=self.reducer_ratings_by_movieID),
            MRStep(mapper=self.mapper_flip_order,
                   mapper_init = self.load_movie_names,
                   reducer=self.reducer_output_sort)]

    def mapper_parse_input(self, key, line):
        # Outputs userID => (movieID, rating)
        (userID, movieID, rating, timestamp) = line.split('\t')
        yield  movieID,  float(rating)

    def reducer_ratings_by_movieID(self, movieID, movieRatings):
        #reduce data to contain only those movieids that have more than 100 ratings
        total = 0
        numRating=0
        for movieRating in movieRatings:
            total += movieRating
            numRating += 1
        if(numRating >100):
            yield movieID, (total/numRating, numRating)
                #use following to make sure getting correct outputs
                #yield (movieID, numRating), total/count 

    def mapper_flip_order(self, movieID, combo):
        avgrating = combo[0]
        count = combo[1]
        sortedrating = '%04.02f'%float(avgrating)
        yield sortedrating, (self.movieNames[int(movieID)],count)

    def reducer_output_sort(self, sortedrating, rating_pairs):
        for moviename, count in rating_pairs:
            yield moviename, (sortedrating, count)


if __name__ == '__main__':
    MovieSimilarities2.run()
    
#!python teamAssignment1.py --items=ml-100k/u.item ml-100k/u.data > movieRatings.txt  
