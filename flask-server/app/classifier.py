from threading import Lock
import joblib
import re
from itertools import islice
import collections

import matplotlib.pyplot as plt

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances = {}

    _lock: Lock = Lock()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field
            # is already initialized, the thread won't create a new object.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


COMMENTS_VOLUME = 1500

class ClassiferSingleton(metaclass=SingletonMeta):
    comment_scraper = YoutubeCommentDownloader()

    def set_paths(self, model_path: str, vectorizer_path: str) -> None:
        self.model = joblib.load(filename=model_path)
        self.vectorizer = joblib.load(filename=vectorizer_path)

    def make_analysis(self, video_url):
        comments_generator = self.comment_scraper.get_comments_from_url(
            youtube_url=video_url, sort_by=SORT_BY_POPULAR)
        
        dirty_comments = [comment['text'] for comment in islice(comments_generator, COMMENTS_VOLUME)]
        clean_comments = self._clean(dirty_comments)
        features = self.vectorizer.transform(clean_comments).toarray()
        predictions = self.model.predict(features)
        counter = collections.Counter(predictions)

        plt.bar(
            x=['Negative', 'Neutral', 'Positive'], 
            height=[counter[0]/COMMENTS_VOLUME*100, counter[1]/COMMENTS_VOLUME*100, counter[2]/COMMENTS_VOLUME*100], 
            color=['red', 'yellow', 'green']
        )
        plt.title(f'Comments Sentiment (Volume:{COMMENTS_VOLUME} instances)')
        plt.xlabel('Category')
        plt.grid(axis='y', zorder=0)
        plt.ylim(0, 100)
        plt.ylabel('% of comments')
        plt.savefig('/app/app/static/plot.png')

        return predictions

    def _clean(self, comments):
        processed_features = []
        for comment in comments:
            # Remove all the special characters
            processed_feature = re.sub(r'\W', ' ', str(comment))
            # remove all single characters
            processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
            # Remove single characters from the start
            processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
            # Substituting multiple spaces with single space
            processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
            # Removing prefixed 'b'
            processed_feature = re.sub(r'^b\s+', '', processed_feature)
            # Converting to Lowercase
            processed_feature = processed_feature.lower()
            processed_features.append(processed_feature)
        return processed_features