<template>
  <div class="h-100">
    <div id="ace"/>
    <div class="save-banner">
      <div>
        <UIButton text="Cancel" @click="leaveCodeVault"/>
      </div>
      <div>
        <UIButton text="Save" :primary="true" @click="save"/>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue, Watch } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import Ace from 'brace';
import 'brace/mode/python';
import '@/assets/ivann-theme';
import { Getter, Mutation } from 'vuex-class';
import Custom from '@/nodes/model/custom/Custom';
import UIButton from '@/components/buttons/UIButton.vue';

@Component({
  components: {
    UIButton,
    Tab,
    Tabs,
  },
})
export default class IdeTab extends Vue {
  @Getter('nodeTriggeringCodeVault') nodeTriggeringCodeVault?: Custom;
  @Getter('inCodeVault') inCodeVault!: boolean;
  @Mutation('leaveCodeVault') leaveCodeVault!: () => void;
  private editor?: Ace.Editor;

  /**
   * Watches the rendering of the CodeVault in order to update code.
   * If the CodeVault was entered from a Custom Node, update code.
   * Else, set empty the editor.
   */
  @Watch('inCodeVault')
  onInCodeVaultChanged(inCodeVault: boolean) {
    if (inCodeVault && this.editor) {
      if (this.nodeTriggeringCodeVault) {
        this.editor.setValue(this.nodeTriggeringCodeVault.getInlineCode());
      } else {
        this.editor.setValue('');
      }
      this.editor.clearSelection();
    }
  }

  mounted() {
    this.editor = Ace.edit('ace');
    this.editor.getSession().setMode('ace/mode/python');
    this.editor.setTheme('ace/theme/ivann');
    this.editor.resize(true);
    if (this.nodeTriggeringCodeVault) {
      this.editor.setValue(this.nodeTriggeringCodeVault.getInlineCode());
      this.editor.clearSelection();
    }
  }

  save() {
    if (this.nodeTriggeringCodeVault && this.editor) {
      this.nodeTriggeringCodeVault.setInlineCode(this.editor.getValue());
    }
    this.leaveCodeVault();
  }
}
</script>

<style scoped>
.save-banner {
  border-top: 1px solid var(--grey);
  background: var(--background);
  height: 3em;
  display: flex;
  padding-left: calc(100% - 11em);
}
.button {
  margin-top: 14px;
  margin-right: 1rem;
}
#ace {
  border-top: 1px solid var(--grey);
  height: calc(100% - 3em);
  font-size: 1em;
  font-family: monospace;
  font-weight: lighter;
}
</style>
