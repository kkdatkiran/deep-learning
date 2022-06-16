package in.dljava.gnotebook.book.storage;

import in.dljava.gnotebook.book.storage.BookPage.BookPageType;
import javafx.application.Platform;
import javafx.beans.property.SimpleListProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;

public class BookManager {

	private SimpleStringProperty bookMessage = new SimpleStringProperty();
	private SimpleListProperty<BookPage> pages = new SimpleListProperty<>(
			FXCollections.observableArrayList(new BookPage(BookPageType.CODE)));

	public BookManager() {

		bookMessage.set("Groovy Notebook");
	}

	public SimpleStringProperty getBookMessageProperty() {
		return this.bookMessage;
	}

	public SimpleListProperty<BookPage> getPagesProperty() {
		return this.pages;
	}

	public void closeGroovyBook() {
		Platform.exit();
	}

	public void addBookPage(BookPageType type, int index) {

		if (index == -1)
			pages.get().add(new BookPage(type));
		else
			pages.get().add(index, null);
	}
}
