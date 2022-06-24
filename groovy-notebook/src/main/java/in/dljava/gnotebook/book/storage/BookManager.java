package in.dljava.gnotebook.book.storage;

import groovy.lang.Binding;
import groovy.lang.GroovyShell;
import in.dljava.gnotebook.book.storage.BookPage.BookPageType;
import javafx.application.Platform;
import javafx.beans.property.SimpleListProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;

public class BookManager {

	private SimpleStringProperty bookMessage = new SimpleStringProperty();
	private SimpleListProperty<BookPage> pages = new SimpleListProperty<>(
			FXCollections.observableArrayList(new BookPage(BookPageType.CODE)));

	private GroovyShell shell;
	private Binding binding;

	public BookManager() {

		bookMessage.set("Groovy Notebook");
		shell = new GroovyShell();
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

	public void addBookPage(BookPageType type, String id) {

		if (id == null)
			pages.get().add(new BookPage(type));
		else {

			int i;
			for (i = 0; i < pages.size(); i++) {

				if (pages.get().get(i).getId().equals(id))
					break;
			}

			if (i == pages.size())
				pages.get().add(new BookPage(type));
			else
				pages.get().add(i, new BookPage(type));
		}
	}

	public void deleteBookPage(String id) {

		pages.removeIf(e -> e.getId().equals(id));
	}

	public void runPage(String id) {

		var optPage = this.pages.getValue().stream().filter(e -> e.getId().equals(id)).findFirst();

		if (optPage.isEmpty())
			return;

		var page = optPage.get();
		if (page.getBookPageTypeProperty().get() == BookPageType.CODE) {

			try {

				if (binding == null) {
					var svr = shell.parse(page.getContentProperty().get());
					svr.run();
					binding = shell.getContext();
					System.out.println("Printing shell context...");
					binding.getVariables().entrySet().stream().forEach(System.out::println);
					System.out.println("Printing script binding context...");
					svr.getBinding().getVariables().entrySet().stream().forEach(System.out::println);
				} else {
					var svr = shell.parse(page.getContentProperty().get());
					svr.setBinding(binding);
					svr.run();
					binding = shell.getContext();
				}
			} catch (Exception ex) {
				page.getOutputProperty().set(ex.toString());
				ex.printStackTrace();
			}
		} else {

		}
	}
}
