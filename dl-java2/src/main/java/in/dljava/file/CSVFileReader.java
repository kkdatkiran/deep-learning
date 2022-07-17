package in.dljava.file;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import in.dljava.DLException;

public class CSVFileReader {

	public static List<List<String>> readCSVFile(String fileName) {
		try (FileReader fr = new FileReader(fileName)) {

			List<List<String>> lines = new ArrayList<>();

			int c;

			int dCount = 0;

			boolean firstChar = true;

			StringBuilder sb = new StringBuilder();
			while ((c = fr.read()) != -1) {

				// Recognising BOM Character
				if (firstChar && (c == 0xFEFF || c == 0xEFBBBF || c == 0xFFFE || c == 0xFFFE0000)) {
					firstChar = false;
					continue;
				}

				if (c == '\n' && ((dCount & 1) == 0)) {
					lines.add(process(sb.toString()));
					sb.setLength(0);
				} else {
					sb.append((char) c);
					if (c == '"') {
						++dCount;
					}
				}
			}

			if (sb.length() != 0) {
				lines.add(process(sb.toString()));
			}

			return lines;
		} catch (IOException ex) {
			ex.printStackTrace();
			throw new DLException("Unable to read file");
		}
	}

	public static List<String> process(String line) {

		int length = line.length();
		if (length == 0)
			return List.of();


		ArrayList<String> list = new ArrayList<>();
		StringBuilder sb = new StringBuilder();
		boolean inDQ = false;
		int i = 0;
		while ((i + 1) < length) {
			char ch = line.charAt(i);
			if (ch == ',' && !inDQ) {
				list.add(sb.toString());
				sb = new StringBuilder();
			} else if (ch == '"') {
				int j = i;
				while ((line.charAt(j) == '"') && (j + 1 < length))
					j++;
				if (((j - i) & 1) == 1)
					inDQ = !inDQ;
				if (j - i != 1) {
					for (int k = 0; k < ((j - i) / 2); k++)
						sb.append('"');
					i = j - 1;
				}
			} else
				sb.append(ch);
			i++;
		}
		sb.append(line.charAt(i) != '\n' && line.charAt(i) != '\r' ? line.charAt(i) : "");
		if (sb.length() > 0)
			list.add(sb.toString());

		return list;
	}

	private CSVFileReader() {

	}
}