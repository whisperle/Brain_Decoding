class SubjectBatchSampler:
    def __iter__(self):
        # Ensure all batches have exactly batch_size samples
        for idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[idx:idx + self.batch_size]
            if len(batch_indices) == self.batch_size:  # Only yield complete batches
                yield batch_indices 