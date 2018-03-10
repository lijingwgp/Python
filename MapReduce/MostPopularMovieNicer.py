from mrjob.job import MRJob
from mrjob.step import MRStep
from io import open

class MostPopularMovie(MRJob):

    def configure_args(self):
        super(MostPopularMovie, self).configure_args()
        self.add_file_arg('--items', help='Path to u.item')

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_ratings,
                   reducer_init=self.reducer_init,
                   reducer=self.reducer_count_ratings),
            MRStep(mapper = self.mapper_passthrough,
                   reducer = self.reducer_find_max)
        ]

    def mapper_get_ratings(self, _, line):
        (userID, movieID, rating, timestamp) = line.split('\t')
        yield movieID, 1

    def reducer_init(self):
        self.movieNames = {}

        with open("u.ITEM", encoding='ascii', errors='ignore') as f:
            for line in f:
                fields = line.split('|')
                self.movieNames[fields[0]] = fields[1]

    def reducer_count_ratings(self, key, values):
        yield None, (sum(values), self.movieNames[key])

#This mapper does nothing; it's just here to avoid a bug in some
#versions of mrjob related to "non-script steps." Normally this
#wouldn't be needed.
    def mapper_passthrough(self, key, value):
        yield key, value

    def reducer_find_max(self, key, values):
        yield max(values)

if __name__ == '__main__':
    MostPopularMovie.run()

# Note, Lines 6-8 may work with older versions with the following
# (in the place of lines 6-8):
#    def configure_options(self):
#        super(MostPopularMovie, self).configure_options()
#        self.add_file_option('--items', help='Path to u.item')
#
# Basically "configure_options" is not supported anymore, so we need
# to use "configure_args" instead.
#
#assumes ml-100k is in same folder as python file and working directory
#Execution Statement
#!python MostPopularMovieNicer.py --items=ml-100k/u.ITEM ml-100k/u.data