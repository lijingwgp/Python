from mrjob.job import MRJob
from mrjob.step import MRStep
from io import open

class MostPopularSuperHero(MRJob):

    def configure_args(self):
        super(MostPopularSuperHero, self).configure_args()
        self.add_file_arg('--names', help='Path to Marvelnames.txt')

    def steps(self):
        return [
            MRStep(mapper=self.mapper_count_friends_per_line,
                   reducer=self.reducer_combine_friends),
            MRStep(mapper=self.mapper_prep_for_sort,
                   mapper_init=self.load_name_dictionary,
                   reducer = self.reducer_find_max_friends)
        ]

    def mapper_count_friends_per_line(self, _, line):
        fields = line.split()
        heroID = fields[0]
        numFriends = len(fields) - 1
        yield int(heroID), int(numFriends)

    def reducer_combine_friends(self, heroID, friendCounts):
        yield heroID, sum(friendCounts)

    def mapper_prep_for_sort(self, heroID, friendCounts):
        heroName = self.heroNames[heroID]
        yield None, (friendCounts, heroName)

    def reducer_find_max_friends(self, key, value):
        yield max(value)

    def load_name_dictionary(self):
        self.heroNames = {}

        with open("Marvelnames.txt", encoding='ascii', errors='ignore') as f:
            for line in f:
                fields = line.split('"')
                heroID = int(fields[0])
                self.heroNames[heroID] = fields[1]

if __name__ == '__main__':
    MostPopularSuperHero.run()

# Note, Lines 6-8 may work with older versions with the following
# (in the place of lines 6-8):
#    def configure_options(self):
#        super(MostPopularSuperHero, self).configure_options()
#        self.add_file_option('--names', help='Path to Marvelnames.txt')
#
# Basically "configure_options" is not supported anymore, so we need
# to use "configure_args" instead.
#
# Execution:
#!python MostPopularSuperhero.py --names=MarvelNames.txt MarvelGraph.txt