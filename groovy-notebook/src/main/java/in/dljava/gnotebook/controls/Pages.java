package in.dljava.gnotebook.controls;

import in.dljava.gnotebook.book.storage.BookManager;
import in.dljava.gnotebook.book.storage.BookPage;
import javafx.collections.ListChangeListener;
import javafx.collections.ObservableList;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.FlowPane;

public class Pages extends ScrollPane { // NOSONAR - This class can
										// have as many parents it
										// want.

	private final BookManager bookManager;

	private final FlowPane wrapperContainer = new FlowPane();
	private final FlowPane container = new FlowPane();

	private final AddBlockBar lastBar;

	public Pages(BookManager bookManager) {

		this.setId("pages");

		this.bookManager = bookManager;
		this.setVbarPolicy(ScrollBarPolicy.ALWAYS);
		this.setHbarPolicy(ScrollBarPolicy.NEVER);

		this.lastBar = new AddBlockBar(bookManager, null);
		this.lastBar.setId("lastBar");

		wrapperContainer.minWidthProperty().bind(this.widthProperty());

		this.container.minWidthProperty().bind(wrapperContainer.widthProperty());
		this.container.getStyleClass().add("pagesContainer");

		this.lastBar.minWidthProperty().bind(wrapperContainer.widthProperty());

		wrapperContainer.getChildren().add(this.container);
		wrapperContainer.getChildren().add(this.lastBar);

		this.setContent(wrapperContainer);
		this.bookManager.getPagesProperty().addListener(this::listChanged);

		this.getStyleClass().add("pages");

		this.initializePages(bookManager.getPagesProperty().get());
	}

	private void initializePages(ObservableList<? extends BookPage> list) {

		this.container.getChildren().clear();

		for (int i = 0; i < list.size(); i++) {
			Page page = new Page(bookManager, list.get(i));
			page.setId(list.get(i).getId());
			page.minWidthProperty().bind(this.container.widthProperty());
			this.container.getChildren().add(page);
		}
	}

	private void listChanged(ListChangeListener.Change<? extends BookPage> page) {
		this.initializePages(page.getList());
	}
}
