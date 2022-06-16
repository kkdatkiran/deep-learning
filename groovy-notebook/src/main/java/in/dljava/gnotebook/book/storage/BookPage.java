package in.dljava.gnotebook.book.storage;

import java.util.UUID;

import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;

public class BookPage {
	
	private String id;
	private SimpleStringProperty content = new SimpleStringProperty();
	private SimpleStringProperty output = new SimpleStringProperty();
	private SimpleObjectProperty<BookPageType> type = new SimpleObjectProperty<>();
	
	public BookPage(BookPageType type) {
		this.id = UUID.randomUUID().toString();
		this.type.set(type);
	}
	
	public BookPage(String id, BookPageType type, String content, String output) {
		this.id = id;
		this.type.setValue(type);
		this.content.setValue(content);
		this.output.setValue(output == null ? "" : output);
	}
	
	public String getId() {
		return this.id;
	}
	
	public SimpleStringProperty getContentProperty() {
		return this.content;
	}
	
	public SimpleStringProperty getOutputProperty() {
		return this.output;
	}
	
	public SimpleObjectProperty<BookPageType> getBookPageTypeProperty() {
		return this.type;
	}

	public enum BookPageType {
		CODE, MARKDOWN
	}
}
