package in.dljava.file;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

import in.dljava.DLException;
import in.dljava.data.DoubleData;
import in.dljava.data.Shape;

public class MNISTReader {

	public static DoubleData readImages(String fileName) {

		try (RandomAccessFile raf = new RandomAccessFile(fileName, "r"); FileChannel channel = raf.getChannel();) {
			int fileSize = (int) channel.size();
			ByteBuffer buffer = ByteBuffer.allocate(fileSize);
			channel.read(buffer);
			buffer.flip();

			int i = 0;
			int magicNumber = 0;
			for (; i < 4; i++) {
				magicNumber <<= 8;
				magicNumber |= Byte.toUnsignedInt(buffer.get());
			}

			int numberOfImages = 0;
			for (; i < 8; i++) {
				numberOfImages <<= 8;
				numberOfImages |= Byte.toUnsignedInt(buffer.get());
			}

			int rows = 0;
			for (; i < 12; i++) {
				rows <<= 8;
				rows |= Byte.toUnsignedInt(buffer.get());
			}

			int cols = 0;
			for (; i < 16; i++) {
				cols <<= 8;
				cols |= Byte.toUnsignedInt(buffer.get());
			}

			if (magicNumber != 2051)
				throw new DLException("File is corrupted");

			double[] data = new double[numberOfImages * rows * cols];

			for (int c = 0; i < fileSize; i++, c++)
				data[c] = buffer.get() == 0 ? 0d : 1d;

			return new DoubleData(new Shape(numberOfImages, rows, cols), data);

		} catch (IOException ex) {
			throw new DLException("Unable to read file.");
		}
	}

	public static DoubleData readLabels(String fileName) {

		try (RandomAccessFile raf = new RandomAccessFile(fileName, "r"); FileChannel channel = raf.getChannel();) {
			int fileSize = (int) channel.size();
			ByteBuffer buffer = ByteBuffer.allocate(fileSize);
			channel.read(buffer);
			buffer.flip();

			int i = 0;
			int magicNumber = 0;
			for (; i < 4; i++) {
				magicNumber <<= 8;
				magicNumber |= Byte.toUnsignedInt(buffer.get());
			}

			int numberOfLabels = 0;
			for (; i < 8; i++) {
				numberOfLabels <<= 8;
				numberOfLabels |= Byte.toUnsignedInt(buffer.get());
			}

			if (magicNumber != 2049)
				throw new DLException("File is corrupted");

			double[] data = new double[numberOfLabels * 10];

			for (int c = 0; i < fileSize; i++, c += 10)
				data[c + buffer.get()] = 1d;

			return new DoubleData(new Shape(numberOfLabels, 10), data);

		} catch (IOException ex) {
			throw new DLException("Unable to read file.");
		}
	}
}
