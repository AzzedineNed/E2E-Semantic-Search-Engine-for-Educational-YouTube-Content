from ETL import processAllChannels
import time
import datetime

print("Starting Multi-Channel Data Pipeline")
print("Time:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 60)

start_time = time.time()
processAllChannels()
end_time = time.time()

total_time = end_time - start_time
print("\n" + "=" * 60)
print("✅ Pipeline completed!")
print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")